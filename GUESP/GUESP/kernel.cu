
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Function to read the .crs file
unordered_map<int, int> readCrsFile(const string& filename) {
    unordered_map<int, int> exams;
    ifstream file(filename);

    if (!file) {
        cerr << "Error: Could not open " << filename << endl;
        exit(1);
    }

    int examID, numStudents;
    while (file >> examID >> numStudents) {
        exams[examID] = numStudents;
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

// Cantor pairing function to generate unique keys for edges
inline int cantorPair(int a, int b) {
    if (a < b) swap(a, b); //ensure a = max & b = min
    return (a + b) * (a + b + 1) / 2 + b;
}

// Function to create the adjacency list and edge map
void createGraphStructures(const vector<vector<int>>& studentEnrollments, unordered_map<int, unordered_set<int>>& adjacencyList, unordered_map<int, int>& edgeMap) {

    for (const auto& enrolledExams : studentEnrollments) {
        for (size_t i = 0; i < enrolledExams.size(); ++i) {
            for (size_t j = i + 1; j < enrolledExams.size(); ++j) {
                int exam1 = enrolledExams[i];
                int exam2 = enrolledExams[j];

                // Add to adjacency list
                adjacencyList[exam1].insert(exam2);
                adjacencyList[exam2].insert(exam1);

                // Add/update edge weight in edgeMap
                int edgeKey = cantorPair(exam1, exam2);
                edgeMap[edgeKey]++;
            }
        }
    }
}

// Function to print the adjacency list
void printAdjacencyList(const unordered_map<int, unordered_set<int>>& adjList) {
    cout << "\nAdjacency List (Exam Connections):\n";
    for (const auto& entry : adjList) {
        cout << "Exam " << entry.first << " -> ";
        for (int neighbor : entry.second) {
            cout << neighbor << " ";
        }
        cout << endl;
    }
}

// Function to print the edge map
void printEdgeMap(const unordered_map<int, int>& edgeMap) {
    cout << "\nEdge Map (Conflicting Students per Exam Pair):\n";
    for (const auto& entry : edgeMap) {
        cout << "Edge Key: " << entry.first << ", Conflicts: " << entry.second << endl;
    }
}

// Function to create an adjacency list
unordered_map<int, unordered_set<int>> createAdjacencyList(const vector<vector<int>>& studentEnrollments) {
    unordered_map<int, unordered_set<int>> adjacencyList;

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

    return adjacencyList;
}

int main()
{
    //filenames of datasets
    string crsFilename = "Data/ute-s-92.crs";
    string stuFilename = "Data/ute-s-92.stu";

    //read data of both files
    unordered_map<int, int> exams = readCrsFile(crsFilename);
    vector<vector<int>> students = readStuFile(stuFilename);

    // Create vairables for adjacency list and edgemap
    unordered_map<int, unordered_set<int>> adjacencyList;
    unordered_map<int, int> edgeMap;

    // Build graph structures
    createGraphStructures(students, adjacencyList, edgeMap);

    // Print results
    printAdjacencyList(adjacencyList);
    printEdgeMap(edgeMap);

    return 0;
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