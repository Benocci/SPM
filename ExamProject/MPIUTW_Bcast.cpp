#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <mpi.h>

using namespace std;

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

    MPI_Init(&argc, &argv); // Initialize the MPI environment

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get the total number of processes

    if (argc != 1 && argc != 2 && argc != 3 ) {
        printf("use: %s N p\n", argv[0]);
        printf("     N size of the square matrix\n");
        printf("     p if 1 print the matrix, if 2 print last value (optional)\n");
        MPI_Finalize();
        return -1;
    }
    // Parse command-line arguments for matrix size and printing option
    if (argc > 1) {
        N = stol(argv[1]);
        if(argc > 2){
            print = stol(argv[2]);
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

   
    double start = MPI_Wtime(); // Measure the start time for the wavefront computation
    
    // Perform the wavefront computation on the matrix:

    for (uint64_t k = 1; k < N; ++k) {
        // Distribute the work among processes, each process handles a subset of rows
        for (uint64_t i = rank; i < N - k; i += size) {
            double dotProduct = 0.0;
            // Calculate the dot product of elements with a row-by-row computation
            for (uint64_t j = 1; j < k + 1; ++j) {
                dotProduct += M[i][i+k - j] * M[i+k][i+j];
            }
            // Assign the cube root of the dot product to the matrix element
            M[i][i+k] = cbrt(dotProduct);
            // Assign the same value to the transposed element
            M[i+k][i] = M[i][i+k];
        }

        // Synchronize all processes to ensure the matrix is updated before moving to the next iteration
        for (uint64_t i = 0; i < N - k; ++i) {
            MPI_Bcast(&M[i][i+k], 1, MPI_DOUBLE, i % size, MPI_COMM_WORLD);  // Broadcast updated matrix elements
            M[i+k][i] = M[i][i+k];  // Ensure symmetry in the matrix
        }
    }

    double end = MPI_Wtime(); // Measure the end time for the wavefront computation

    // Process with rank 0 prints the elapsed time and optionally the matrix or a specific value
    if (rank == 0) {
        std::cout << "Time with " << size << " processes: " << end-start << " seconds" << std::endl;
        if(print == 1){
            printMatrix(M,N);
        }
        else if(print == 2){
            printf("Last value [0][%ld]=%f\n",N-1, M[0][N-1]);
        }
    }

    MPI_Finalize(); // Finalize the MPI environment

    return 0;
}
