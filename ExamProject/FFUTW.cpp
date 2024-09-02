#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <ff/parallel_for.hpp>
#include <hpc_helpers.hpp>

#define MAX_THREAD 40  // Define the maximum number of threads
using namespace std;
using namespace ff;

// Function to perform the wavefront computation using parallel processing
void wavefront(
    vector<vector<double>>& M,
    const uint64_t &N,
    int numThreads) {
    ParallelFor pf(numThreads, true, true);  // Initialize a ParallelFor object with the specified number of threads

    // Traverse through the matrix diagonally in a wavefront manner
	for (uint64_t k = 1; k < N; ++k) {
        // Use parallel processing to iterate over the elements along the current diagonal
        pf.parallel_for(0, N-k, 1, [&, N, k](const int i) {
            double dotProduct = 0.0;
            // Calculate the dot product of elements with row-by-column computation
            for (uint64_t j = 1; j < k + 1; ++j) {
                dotProduct += M[i][i + k - j] * M[i + j][i + k];
            }
            // Assign the cube root of the dot product to the matrix element
            M[i][i + k] = cbrt(dotProduct);
        }, numThreads);// Specify the number of threads to be used in the parallel loop
    }
}

// Function to print the NxN matrix M with fixed precision
void printMatrix(const vector<vector<double>>& M, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << setw(10) << fixed << setprecision(7) << M[i][j] << " ";
        }
        cout << endl;
    }
}


int main(int argc, char *argv[]) {
	uint64_t N = 6;          // default size of the matrix (NxN)
    uint64_t numThreads = 4; // default number of threads
    uint64_t print = 0;      // Variable to control printing options

    if (argc < 1 || argc > 4) {
		printf("use: %s N numThreads p\n", argv[0]);
		printf("     N size of the square matrix\n");	
		printf("     numThread number of thread (optional)\n");
        printf("     p if 1 print the matrix (optional)\n");	
		return -1;
	}
    // Parse command-line arguments for matrix size, number of threads, and printing option
	if (argc > 1) {
		N = stol(argv[1]);
        if(argc > 2){
            numThreads = stol(argv[2]);
            if(argc > 3){
                print = stol(argv[3]);
            }
        }
	}

	// Allocate and initialize the matrix M with zeros
	vector<vector<double>> M(N, vector<double>(N, 0.0));

    // Initialize the diagonal elements of the matrix
    auto init=[&]() {
		for(uint64_t i=0; i < N; ++i){
		    M[i][i] = static_cast<double>(i + 1) / N;
	    }
	};

    init();

    ffTime(START_TIME);          // Start measuring the time for the wavefront computation
	wavefront(M, N, numThreads); // Perform the wavefront computation on the matrix using the specified number of threads
    ffTime(STOP_TIME);           // Stop measuring the time

    std::cout << "# elapsed time (wavefront): " << ffTime(GET_TIME)/1000  << "s" << std::endl;

    if(print == 1){
        printMatrix(M,N);
    }
    else if(print == 2){
        printf("Last value [0][%ld]=%f\n",N-1, M[0][N-1]);
    }

    return 0;
}
