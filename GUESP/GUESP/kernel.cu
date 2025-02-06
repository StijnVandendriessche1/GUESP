#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <iostream>
#include <sstream>
#include <queue>


#define DEBUG false
#define DBTID 9


#define MAX_COLORS 10
#define NUM_NODES 184
#define NUM_EDGES 1430
#define NUM_CORES 3072
#define NUM_BLOCKS 48


#define MOVE_SIZE ((MAX_COLORS - 1) * NUM_NODES)
#define SWAP_SIZE ((NUM_NODES * (NUM_NODES - 1)) / 2)


using namespace std;


//struct to make reduction (minimizing) easier
struct Candidate {
    int score;   // the candidate score (to be minimized)
    int op;      // the operation id that produced that score
    bool move;   // true if it came from a move, false if from a swap
};

// Define constant memory on GPU for adjacency list, and edge map, global memory for solution and score
__constant__ int d_adjList[2 * NUM_EDGES];
__constant__ int d_adjListStartIndices[2 + NUM_NODES];
__constant__ int d_cantorPairs[NUM_EDGES];
__constant__ int d_weights[NUM_EDGES];
int* d_solution; //184 ints for the coloring
int* d_score; //one int for the score
Candidate* d_globalCandidates;





// ---- ---- ---- ---- GPU DEVICE FUNCTIONS

// Cantor pairing function for GPU
__device__ inline int cantorPairGPU(int a, int b) {
    // Ensure a >= b for consistency
    if (a < b) {
        int temp = a;
        a = b;
        b = temp;
    }
    int sum = a + b;
    return (sum * (sum + 1)) / 2 + b;
}

// Binary search for a key in constant memory
__device__ int binarySearchConstant(int key, int size) {
    int left = 0, right = size - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2; // Avoid overflow

        if (d_cantorPairs[mid] == key)
            return d_weights[mid];  // Found key, return value

        if (d_cantorPairs[mid] < key)
            left = mid + 1;
        else
            right = mid - 1;
    }

    return -1; // Key not found
}

// Delta cost function for move exam
__device__ int deltaMoveExam(int tid, int* d_solution)
{
    int pos = tid / (MAX_COLORS - 1);
    int val = (tid % (MAX_COLORS - 1) + d_solution[pos] + 1) % MAX_COLORS;
    int old = d_solution[pos];

    // Compute the delta cost of moving `examID` from `oldColor` to `newColor`
    int deltaCost = 0;
    int P_0 = 0; //Partial penalty for current solution
    int P_0_prime = 0; //Partial penalty for candidate solution

    // Retrieve adjacency list range for this node
    int startIdx = d_adjListStartIndices[pos + 1];
    int endIdx = d_adjListStartIndices[pos + 2];

    if (DEBUG)
    {
        if (tid == DBTID)
        {
            printf("TID: %d\n", tid);
            printf("pos: %d\tval: %d\n", pos, val);
            printf("old: %d\n", old);
            printf("indeces: %d, %d\n", startIdx, endIdx);
            printf("neighbors: \n");
        }
    }

    // Iterate over neighbors and calculate P_0 and P_0'
    for (int i = startIdx; i < endIdx; i++)
    {
        int neighbor = d_adjList[i];
        int neighborColor = d_solution[neighbor-1];
        if (val == neighborColor) return 1;
        int weight = binarySearchConstant(cantorPairGPU(pos+1, neighbor), NUM_EDGES);
        if (DEBUG)
        {
            if (tid == DBTID)
            {
                printf("%d, color: %d, weight: %d\n", neighbor, neighborColor, weight);
            }
        }

        // Calculate P_0 (Penalty before move)
        if (abs(old - neighborColor) < 6) {
            P_0 += weight * (1 << (5 - abs(old - neighborColor))); // 2^(5-dist)
        }

        // Calculate P_0' (Penalty after move)
        if (abs(val - neighborColor) < 6) {
            P_0_prime += weight * (1 << (5 - abs(val - neighborColor))); // 2^(5-dist)
        }
    }

    // Calculate Delta P_move (Change in penalty)
    deltaCost = P_0_prime - P_0;

    return deltaCost;  // Return change in cost for this move
}

// Delta cost function for swap exam
__device__ int swapExams(int tid, int* d_solution)
{
    int i = tid+1;
    return 100;
}

// simple comparisson function between candidates
__device__ Candidate minCandidate(const Candidate& a, const Candidate& b) 
{
    return (a.score <= b.score) ? a : b;
}





// ---- ---- ---- ---- GPU MAIN KERNELS

// kernel for optimizing the given initial solution
__global__ void optimizeSolutionKernel(int* d_solution, int* d_score, Candidate* d_globalCandidates)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Compute thread index
    int localMin = *d_score+INT_MAX/2+1;  // initial value is the current score
    int bestOp = -1;          // candidate operation id
    bool bestMove = true;     // whether it is a move or a swap

    while (tid < MOVE_SIZE + SWAP_SIZE) {
        int candidate;
        bool candidateMove;
        if (tid < MOVE_SIZE) 
        {
            candidate = deltaMoveExam(tid, d_solution) + *d_score;
            candidateMove = true;
            if (DEBUG)
            {
                if (tid == DBTID)
                {
                    printf("score: %d\n", candidate);
                }
            }
        }
        else {
            candidate = swapExams(tid, d_solution) + *d_score;
            candidateMove = false;
        }
        if (candidate < localMin) {
            localMin = candidate;
            bestOp = tid;
            bestMove = candidateMove;
        }
        tid += 3072;
    }

    // Pack the thread’s best candidate into a structure.
    Candidate myCand;
    myCand.score = localMin;
    myCand.op = bestOp;
    myCand.move = bestMove;

    //--------------------------------------------------------------------------
    // Block-level reduction using shared memory.
    // We assume blockDim.x is 64 (as you set) so we can declare a fixed-size shared array.
    __shared__ Candidate sdata[64];
    int lane = threadIdx.x;
    sdata[lane] = myCand;
    __syncthreads();

    // Reduce the candidates in shared memory.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lane < s) {
            sdata[lane] = minCandidate(sdata[lane], sdata[lane + s]);
        }
        __syncthreads();
    }

    // Copy the minima from the blocks to global memory
    if (lane == 0) 
    {
        d_globalCandidates[blockIdx.x] = sdata[0];
    }
}

// kernel to reduce the minimum values found by all blocks to one overall minimum and put it at d_globalCandidates[0]
__global__ void reduce(Candidate* d_globalCandidates, int* d_solution, int* d_score)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ Candidate sdata[64];
    // Load data from global memory into shared memory
    sdata[idx] = d_globalCandidates[idx];
    __syncthreads();
    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (idx < s) {
            sdata[idx] = minCandidate(sdata[idx], sdata[idx + s]);
        }
        __syncthreads();
    }
    // Write the result from block 0 to global memory
    if (idx == 0) 
    {
        d_globalCandidates[0] = sdata[0];
        Candidate best = sdata[0];
        if (best.move)
        {
            int pos = best.op / (MAX_COLORS - 1);
            int val = (best.op % (MAX_COLORS - 1) + d_solution[pos] + 1) % MAX_COLORS;
            d_solution[pos] = val;
            *d_score = best.score;
        }
        else
        {
            //TODO execute the swap
        }
        printf("iteration completed, current score: %d\n", best.score);
        /*printf("[");
        for (int i = 0; i < NUM_NODES; i++)
        {
            printf("%d, ", d_solution[i]);
        }
        printf("]");*/
    }
}





// ---- ---- ---- ---- COPY DATA TO AND FROM GPU + calling kernels

// Function to copy adjacency list data to GPU constant memory
//TODO: check if adjacencyList fit's in constant memory, if not it can be put in global memory
void copyAdjacencyListToGPU(const vector<int>& adjList, const vector<int>& adjListStartIndices) {
    // Copy adjacency list data to GPU constant memory
    cudaError_t err = cudaMemcpyToSymbol(d_adjList, adjList.data(), adjList.size() * sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA Error AL: %s\n", cudaGetErrorString(err));
    }
    err = cudaMemcpyToSymbol(d_adjListStartIndices, adjListStartIndices.data(), adjListStartIndices.size() * sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA Error ALI: %s\n", cudaGetErrorString(err));
    }
}

// Function to copy sorted key-value pairs to GPU (constant memory)
void copySortedPairsToGPU(const vector<int>& keys, const vector<int>& values) {
    // Ensure the array size is within limits
    size_t size = keys.size() * sizeof(int);

    // Copy sorted keys and values to constant memory
    cudaError_t err = cudaMemcpyToSymbol(d_cantorPairs, keys.data(), size);
    if (err != cudaSuccess) {
        printf("CUDA Error CP: %s\n", cudaGetErrorString(err));
    }
    err = cudaMemcpyToSymbol(d_weights, values.data(), size);
    if (err != cudaSuccess) {
        printf("CUDA Error W: %s\n", cudaGetErrorString(err));
    }
}

// Function to copy edge map to GPU using sorted keys for binary search
void copyEdgeMapToGPU(unordered_map<int, int> edgeMap) 
{
    vector<pair<int, int>> pairs(edgeMap.begin(), edgeMap.end()); // Convert map to vector of pairs

    // Sort pairs by keys
    sort(pairs.begin(), pairs.end(), [](const pair<int, int>& a, const pair<int, int>& b) {
        return a.first < b.first;
        });

    // Separate keys and values into separate vectors
    vector<int> keys, values;
    for (const auto& p : pairs) {
        keys.push_back(p.first);
        values.push_back(p.second);
    }

    // Copy sorted arrays to GPU
    copySortedPairsToGPU(keys, values);
}

// Function to copy initial solution to GPU constant memory
void copySolutionToGPU(const vector<int>& coloring) {
    cudaMalloc((void**)&d_solution, coloring.size() * sizeof(int));
    cudaError_t err = cudaMemcpy(d_solution, coloring.data(), coloring.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Error SOL: %s\n", cudaGetErrorString(err));
    }
}

// Function to copy initial score to GPU constant memory
void copyCostToGPU(const int& score) {
    cudaMalloc((void**)&d_score, sizeof(int));
    cudaError_t err = cudaMemcpy(d_score, &score, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("CUDA Error COST: %s\n", cudaGetErrorString(err));
    }
}

vector<int> copySolutionToCpu(int* d_solution) 
{
    vector<int> h_solution(NUM_NODES);
    cudaMemcpy(h_solution.data(), d_solution, NUM_NODES * sizeof(int), cudaMemcpyDeviceToHost);
    return h_solution;
}

// Function for calling the main GPU kernel
void optimizeSolutionOnGPU(int* d_solution, int* d_score, Candidate* d_globalCandidates, int cost) 
{
    //allocate 48 Candidate structs for the reductions step in the algorithm
    cudaMalloc((void**)&d_globalCandidates, 48 * sizeof(Candidate));

    //given the dimensions of the NVIDIA GeForce RTX 2080 SUPER Max-Q 48 blocks of 64 threads is the best configuration
    int threadsPerBlock = 64;
    int blocksPerGrid = 48;

    int h_oldScore = cost;
    int h_newScore = cost;
    int count = 0;

    do {
        h_oldScore = h_newScore;
        // Launch kernel to find the best candidate
        optimizeSolutionKernel << <blocksPerGrid, threadsPerBlock >> > (d_solution, d_score, d_globalCandidates);
        cudaDeviceSynchronize();

        // Reduce and apply the best move
        reduce << <1, blocksPerGrid >> > (d_globalCandidates, d_solution, d_score);
        cudaDeviceSynchronize();

        // Copy updated score from device to host
        cudaMemcpy(&h_newScore, d_score, sizeof(int), cudaMemcpyDeviceToHost);
    } while (h_newScore < h_oldScore);

    printf("local optimal solution score: %d\n", h_newScore);

    /*// Launch kernel
    optimizeSolutionKernel << <blocksPerGrid, threadsPerBlock >> > (d_solution, d_score, d_globalCandidates);
    cudaDeviceSynchronize();
    reduce << <1, NUM_BLOCKS >> > (d_globalCandidates, d_solution, d_score);
    cudaDeviceSynchronize();*/
    //TODO: copy element zero from d_globalCandidates back and check if the score has changed, if so reïterate over both kernels, if not -> done
}





// ---- ---- ---- ---- READ IN DATA FROM DATASETS

// Function to read the .crs file
unordered_map<int, int> readCrsFile(const string& filename, int& maxExamID) {
    unordered_map<int, int> exams;
    ifstream file(filename);

    if (!file) {
        cerr << "Error: Could not open " << filename << endl;
        exit(1);
    }

    int examID, numStudents;
    maxExamID = 0;
    while (file >> examID >> numStudents) {
        exams[examID] = numStudents;
        maxExamID = max(maxExamID, examID); // Track max exam ID
    }

    file.close();
    return exams;
}

// Function to read the .stu file
vector<vector<int>> readStuFile(const string& filename) {
    vector<vector<int>> studentEnrollments;
    ifstream file(filename);

    if (!file) {
        cerr << "Error: Could not open " << filename << endl;
        exit(1);
    }

    string line;
    while (getline(file, line)) {
        vector<int> exams;
        stringstream ss(line);
        int examID;

        while (ss >> examID) {
            exams.push_back(examID);
        }

        studentEnrollments.push_back(exams);
    }

    file.close();
    return studentEnrollments;
}





// ---- ---- ---- ---- CREATE GRAPH DATA STRUCTURES AND RELEVANT FUNCTIONS

// Function to create an adjacency list
void createAdjacencyList(const vector<vector<int>>& studentEnrollments, int max, vector<int>& adjList, vector<int>& startIndeces)
{
    unordered_map<int, unordered_set<int>> adjacencyList; //temp adj list

    for (const auto& enrolledExams : studentEnrollments) {
        // Create a connection between all pairs of exams in this student's enrollment
        for (size_t i = 0; i < enrolledExams.size(); ++i) {
            for (size_t j = i + 1; j < enrolledExams.size(); ++j) {
                int exam1 = enrolledExams[i];
                int exam2 = enrolledExams[j];

                adjacencyList[exam1].insert(exam2);
                adjacencyList[exam2].insert(exam1);
            }
        }
    }

    // Flatten adjacency list into arrays
    startIndeces.resize(max+2, 0);
    startIndeces[0] = adjacencyList.size();
    for (int i = 1; i < max+1; ++i) 
    {
        startIndeces[i] = adjList.size();
        adjList.insert(adjList.end(), adjacencyList[i].begin(), adjacencyList[i].end());
    }
    startIndeces[max+1] = adjList.size();
}

// Function to get the neighbors of a given exam
vector<int> getNeighbors(int examID, const vector<int>& adjList, const vector<int>& startIndeces) 
{
    if (examID < 1 || examID > startIndeces[0]) 
    {
        cerr << "Error: Invalid exam ID " << examID << endl;
        return {};
    }

    // Get neighbors range from startIndeces
    int start = startIndeces[examID];
    int end = startIndeces[examID + 1];

    // Extract neighbors from adjList
    vector<int> neighbors(adjList.begin() + start, adjList.begin() + end);
    return neighbors;
}

// Cantor pairing function
inline int cantorPair(int a, int b) {
    if (a < b) swap(a, b); // Ensure a > b for consistency
    return (a + b) * (a + b + 1) / 2 + b;
}

// Inverse Cantor pairing function
pair<int, int> inverseCantor(int Z) {
    int w = floor((sqrt(8.0 * Z + 1) - 1) / 2); // Solve for w
    int t = (w * (w + 1)) / 2;                  // Compute triangular number
    int b = Z - t;
    int a = w - b;
    return {a, b};
}

// Function to create the edge map using Cantor pairing
void createEdgeMap(const vector<vector<int>>& studentEnrollments, unordered_map<int, int>& edgeMap) {
    for (const auto& exams : studentEnrollments) {
        for (size_t i = 0; i < exams.size(); ++i) {
            for (size_t j = i + 1; j < exams.size(); ++j) {
                int exam1 = exams[i];
                int exam2 = exams[j];

                int key = cantorPair(exam1, exam2); // Compute unique key using Cantor pairing

                // Increment edge weight
                edgeMap[key] += 1;
            }
        }
    }
}

// Function to get the weight of the edge between two exams
int getWeight(int exam1, int exam2, const unordered_map<int, int>& edgeMap) {
    int key = cantorPair(exam1, exam2); // Compute unique key using Cantor pairing

    // Check if the edge exists in the map
    if (edgeMap.find(key) != edgeMap.end()) {
        return edgeMap.at(key); // Return the weight
    }
    return 0; // Return 0 if no edge exists
}





// ---- ---- ---- ---- INITIAL SOLUTION DSATUR SOLVER

// Function to find the smallest available color
int findSmallestAvailableColor(const unordered_set<int>& neighborColors) {
    int color = 0;
    while (neighborColors.find(color) != neighborColors.end()) {
        color++;
    }
    return color;
}

// DSATUR algorithm implementation
vector<int> dsaturColoring(const vector<int>& adjList, const vector<int>& adjListStartIndices) 
{
    int numNodes = adjListStartIndices[0];
    vector<int> coloring(numNodes, -1); // -1 indicates uncolored
    vector<int> saturation(numNodes, 0); // Saturation degree for each node
    vector<int> degrees(numNodes, 0); // Degree of each node

    // Calculate degrees of each node
    for (int i = 1; i <= numNodes; ++i) {
        degrees[i - 1] = adjListStartIndices[i + 1] - adjListStartIndices[i];
    }

    // DSATUR algorithm loop
    while (count(coloring.begin(), coloring.end(), -1) > 0) {
        // Find the uncolored node with the largest saturation degree, break ties by degree
        int maxNode = -1;
        int maxSaturation = -1;
        int maxDegree = -1;

        for (int i = 0; i < numNodes; ++i) {
            if (coloring[i] == -1) { // Node is uncolored
                if (saturation[i] > maxSaturation ||
                    (saturation[i] == maxSaturation && degrees[i] > maxDegree)) {
                    maxNode = i;
                    maxSaturation = saturation[i];
                    maxDegree = degrees[i];
                }
            }
        }

        // Find the smallest available color for the selected node
        unordered_set<int> neighborColors;
        for (int j = adjListStartIndices[maxNode + 1]; j < adjListStartIndices[maxNode + 2]; ++j) {
            int neighbor = adjList[j];
            if (coloring[neighbor - 1] != -1) {
                neighborColors.insert(coloring[neighbor - 1]);
            }
        }
        int selectedColor = findSmallestAvailableColor(neighborColors);
        coloring[maxNode] = selectedColor;

        // Update saturation degree of uncolored neighbors
        for (int j = adjListStartIndices[maxNode + 1]; j < adjListStartIndices[maxNode + 2]; ++j) {
            int neighbor = adjList[j];
            if (coloring[neighbor - 1] == -1) {
                unordered_set<int> neighborColors;
                for (int k = adjListStartIndices[neighbor]; k < adjListStartIndices[neighbor + 1]; ++k) {
                    int subNeighbor = adjList[k];
                    if (coloring[subNeighbor - 1] != -1) {
                        neighborColors.insert(coloring[subNeighbor - 1]);
                    }
                }
                saturation[neighbor - 1] = neighborColors.size();
            }
        }
    }

    return coloring;
}





// ---- ---- ---- ---- EVALUTION OF SOLUTION

// Function to calculate the cost of a given coloring solution
double calculateCost(const vector<int>& coloring, const unordered_map<int, int>& edgeMap) {
    double totalPenalty = 0.0;

    for (const auto& entry : edgeMap) {
        int key = entry.first;
        int weight = entry.second;  // Number of shared students (w_ij)

        // Decode Cantor-paired key to get the two exams (i, j)
        pair<int, int> exams = inverseCantor(key);
        int exam1 = exams.first;
        int exam2 = exams.second;

        // Get assigned time slots for these exams
        int t_i = coloring[exam1 - 1];
        int t_j = coloring[exam2 - 1];

        // Compute time difference
        int distance = abs(t_i - t_j);

        // Apply the penalty formula only if |t_i - t_j| < 6
        if (distance < 6) {
            totalPenalty += weight * pow(2, 5-distance);
        }
    }

    return totalPenalty;
}


void printSolution(vector<int> coloring, int numNodes) {
    // Print coloring result
    cout << "Graph Coloring:\n";
    for (int i = 0; i < numNodes; ++i) {
        cout << "Node " << i + 1 << ": Color " << coloring[i] << endl;
    }
}





// ---- ---- ---- ---- MAIN

// main
int main()
{
    //READING DATA
    //filenames of datasets
    string crsFilename = "Data/ute-s-92.crs";
    string stuFilename = "Data/ute-s-92.stu";

    //read data of both files
    int maxExamID;
    unordered_map<int, int> exams = readCrsFile(crsFilename, maxExamID);
    vector<vector<int>> students = readStuFile(stuFilename);

    // Define adjacency list storage
    vector<int> adjList;
    vector<int> adjListStartIndices;



    //GRAPH CREATION AND COPY
    // Build the adjacency list
    createAdjacencyList(students, maxExamID, adjList, adjListStartIndices);

    // Copy adjacency list to GPU constant memory
    copyAdjacencyListToGPU(adjList, adjListStartIndices);

    //Define edge map
    unordered_map<int, int> edgeMap;

    //Build the edge map
    createEdgeMap(students, edgeMap);

    // Copy edge map to GPU (sorted for binary search)
    copyEdgeMapToGPU(edgeMap);



    //INITIAL SOLUTION CREATION AND COPY
    // Perform DSATUR graph coloring
    vector<int> coloring = dsaturColoring(adjList, adjListStartIndices);

    // Copy coloring to GPU constant memory
    copySolutionToGPU(coloring);



    //INITIAL EVALUATION AND COPY
    // Calculate cost
    double cost = calculateCost(coloring, edgeMap);

    // Copy the cost to GPU constant memory
    copyCostToGPU(cost);

    //GPU EXECTUION AND COPY BACK
    // Execute kernel to find best solution
    optimizeSolutionOnGPU(d_solution, d_score, d_globalCandidates, cost);

    // Copy solution and score back to CPU
    vector<int> solution = copySolutionToCpu(d_solution);

    //FINAL EVALUATION
    // Check score
    //double newCost = calculateCost(solution, edgeMap);
    
    printf("[");
    for (int i = 0; i < NUM_NODES; i++)
    {
        printf("%d, ", solution[i]);
    }
    printf("]");

    return 0;
}

















































//print functions for debugging
// Function to print the adjacency list
void printAdjacencyList(const vector<int>& adjList, const vector<int>& adjListStartIndices) {
    cout << "\nFlattened Adjacency List:\n";
    for (size_t i = 1; i <= adjListStartIndices[0]; ++i)
    {
        cout << "Exam " << (i) << " -> ";
        for (int j = adjListStartIndices[i]; j < adjListStartIndices[i + 1]; ++j)
        {
            cout << (adjList[j]) << " ";
        }
        cout << endl;
    }
}

// Function to print the edge map
void printEdgeMap(const unordered_map<int, int>& edgeMap) {
    cout << "\nEdge Map (Shared Students Between Exam Pairs):\n";
    for (const auto& entry : edgeMap) {
        int key = entry.first;
        int weight = entry.second;

        // Decode the Cantor pair (Optional: For readability)
        int w = floor((sqrt(8.0 * key + 1) - 1) / 2); // Solve for w
        int t = (w * (w + 1)) / 2;                   // Triangular number
        int exam1 = w - (key - t);
        int exam2 = key - t;

        cout << "Edge (" << exam1 << ", " << exam2 << ") -> Weight: " << weight << endl;
    }
}

// Function to print the adjacency list from the GPU
__global__ void printAdjListFromGPU(int numNodes) {
    int idx = threadIdx.x + 1;  // Start from 1, as index 0 stores the number of nodes

    if (idx <= numNodes) {  // Ensure within valid exam IDs
        int start = d_adjListStartIndices[idx];       // Corrected start index
        int end = d_adjListStartIndices[idx + 1];    // Corrected end index

        // Allocate a buffer in shared memory (adjust size if needed)
        char outputBuffer[512];
        int offset = 0;

        // Manually construct the output string
        outputBuffer[offset++] = 'E';
        outputBuffer[offset++] = 'x';
        outputBuffer[offset++] = 'a';
        outputBuffer[offset++] = 'm';
        outputBuffer[offset++] = ' ';

        // Convert exam ID to characters (idx) and append to buffer
        int examId = idx;
        int tempOffset = offset;
        do {
            outputBuffer[tempOffset++] = '0' + (examId % 10);
            examId /= 10;
        } while (examId > 0);

        // Reverse the digits in place
        for (int j = 0; j < (tempOffset - offset) / 2; ++j) {
            char temp = outputBuffer[offset + j];
            outputBuffer[offset + j] = outputBuffer[tempOffset - 1 - j];
            outputBuffer[tempOffset - 1 - j] = temp;
        }

        offset = tempOffset;
        outputBuffer[offset++] = ' ';
        outputBuffer[offset++] = '-';
        outputBuffer[offset++] = '>';

        // Append neighbor exam IDs to the buffer
        for (int i = start; i < end; ++i) {
            outputBuffer[offset++] = ' ';

            int neighborId = d_adjList[i];  // Corrected: no "+1"
            int tempOffset = offset;
            do {
                outputBuffer[tempOffset++] = '0' + (neighborId % 10);
                neighborId /= 10;
            } while (neighborId > 0);

            // Reverse the digits in place
            for (int j = 0; j < (tempOffset - offset) / 2; ++j) {
                char temp = outputBuffer[offset + j];
                outputBuffer[offset + j] = outputBuffer[tempOffset - 1 - j];
                outputBuffer[tempOffset - 1 - j] = temp;
            }

            offset = tempOffset;
        }

        outputBuffer[offset++] = '\n';
        outputBuffer[offset] = '\0';  // Null terminate the string

        // Single printf() per thread
        printf("%s", outputBuffer);
    }
}

// Function to print the weighted graph from the GPU
__global__ void printGraphWithWeights(int numNodes) {
    int idx = threadIdx.x + 1;  // Start from 1, as index 0 stores the number of nodes

    if (idx <= numNodes) {  // Ensure within valid exam IDs
        int start = d_adjListStartIndices[idx];       // Start index in adjacency list
        int end = d_adjListStartIndices[idx + 1];    // End index in adjacency list

        // Allocate a buffer in shared memory (adjust size if needed)
        char outputBuffer[512];
        int offset = 0;

        // Manually construct the output string
        outputBuffer[offset++] = 'E';
        outputBuffer[offset++] = 'x';
        outputBuffer[offset++] = 'a';
        outputBuffer[offset++] = 'm';
        outputBuffer[offset++] = ' ';

        // Convert exam ID to characters (idx) and append to buffer
        int examId = idx;
        int tempOffset = offset;
        do {
            outputBuffer[tempOffset++] = '0' + (examId % 10);
            examId /= 10;
        } while (examId > 0);

        // Reverse the digits in place
        for (int j = 0; j < (tempOffset - offset) / 2; ++j) {
            char temp = outputBuffer[offset + j];
            outputBuffer[offset + j] = outputBuffer[tempOffset - 1 - j];
            outputBuffer[tempOffset - 1 - j] = temp;
        }

        offset = tempOffset;
        outputBuffer[offset++] = ' ';
        outputBuffer[offset++] = '-';
        outputBuffer[offset++] = '>';

        // Append neighbor exam IDs and their respective weights
        for (int i = start; i < end; ++i) {
            outputBuffer[offset++] = ' ';

            int neighborId = d_adjList[i];  // Get neighbor ID
            int tempOffset = offset;

            // Convert neighbor ID to characters and append
            do {
                outputBuffer[tempOffset++] = '0' + (neighborId % 10);
                neighborId /= 10;
            } while (neighborId > 0);

            // Reverse the digits in place
            for (int j = 0; j < (tempOffset - offset) / 2; ++j) {
                char temp = outputBuffer[offset + j];
                outputBuffer[offset + j] = outputBuffer[tempOffset - 1 - j];
                outputBuffer[tempOffset - 1 - j] = temp;
            }

            offset = tempOffset;

            // Get edge weight using Cantor pairing and binary search
            int weight = binarySearchConstant(cantorPairGPU(idx, d_adjList[i]), NUM_EDGES);

            // Append " (w=" part
            outputBuffer[offset++] = ' ';
            outputBuffer[offset++] = '(';
            outputBuffer[offset++] = 'w';
            outputBuffer[offset++] = '=';

            // Convert weight to characters and append
            tempOffset = offset;
            int weightValue = weight;
            do {
                outputBuffer[tempOffset++] = '0' + (weightValue % 10);
                weightValue /= 10;
            } while (weightValue > 0);

            // Reverse the digits in place
            for (int j = 0; j < (tempOffset - offset) / 2; ++j) {
                char temp = outputBuffer[offset + j];
                outputBuffer[offset + j] = outputBuffer[tempOffset - 1 - j];
                outputBuffer[tempOffset - 1 - j] = temp;
            }

            offset = tempOffset;

            // Append closing bracket
            outputBuffer[offset++] = ')';
        }

        outputBuffer[offset++] = '\n';
        outputBuffer[offset] = '\0';  // Null terminate the string

        // Single printf() per thread
        printf("%s", outputBuffer);
    }
}


















/*
int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <crs_file> <stu_file>\n";
        return 1;
    }

    string crsFilename = argv[1];
    string stuFilename = argv[2];

    unordered_map<int, int> exams = readCrsFile(crsFilename);
    vector<vector<int>> students = readStuFile(stuFilename);

    // Print results (for debugging)
    printDatasets(exams, students);

    return 0;
}
*/