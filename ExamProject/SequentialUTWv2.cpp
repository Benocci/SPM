#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <hpc_helpers.hpp>

using namespace std;

// Function to perform the wavefront computation
void wavefront(
    vector<vector<double>>& M,
    const uint64_t &N) {

    // Traverse through the matrix diagonally
	for (uint64_t k = 1; k < N; ++k) {
        // Loop over the elements along the current diagonal
        for (uint64_t i = 0; i < N - k; ++i) {
            double dotProduct = 0.0;
            // Calculate the dot product of elements with row-by-row computation
            for (uint64_t j = 1; j < k + 1; ++j) {
                dotProduct += M[i][i+k - j] * M[i+k][i+j];
            }
            // Assign the cube root of the dot product to the matrix element
            M[i][i+k] = cbrt(dotProduct);
            // Assign the same value to the transposed element
            M[i+k][i] = M[i][i+k];
        }
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
	uint64_t N = 6;     // default size of the matrix (NxN)
	uint64_t print = 0; // Variable to control printing options

    if (argc != 1 && argc != 2 && argc != 3 ) {
		printf("use: %s N p\n", argv[0]);
		printf("     N size of the square matrix\n");	
		printf("     p if 1 print the matrix, if 2 print last value (optional)\n");	
		return -1;
	}
    // Parse command-line arguments for matrix size and printing option
	if (argc > 1) {
		N = stol(argv[1]);
        if(argc > 2){
            print = stol(argv[2]);
        }
	}

	// Allocate the matrix
	vector<vector<double>> M(N, vector<double>(N, 0.0));

    // Initialize the diagonal elements of the matrix
    auto init=[&]() {
		for(uint64_t i=0; i < N; ++i){
		    M[i][i] = static_cast<double>(i + 1) / N;
	    }
	};

    init();

	TIMERSTART(wavefront);  // Start the timer for the wavefront computation
	wavefront(M, N);        // Perform the wavefront computation on the matrix
    TIMERSTOP(wavefront);   // Stop the timer

    if(print == 1){
        printMatrix(M,N);
    }
    else if(print == 2){
        printf("Last value [0][%ld]=%f\n",N-1, M[0][N-1]);
    }

    return 0;
}
