#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <mpi.h>
#include <hpc_helpers.hpp>

using namespace std;

void wavefront(vector<vector<double>>& M, const uint64_t &N, int rank, int size) {
    for (uint64_t k = 1; k < N; ++k) {
        for (uint64_t i = rank; i < N - k; i += size) {
            double dotProduct = 0.0;
            for (uint64_t j = 0; j < k + 1; ++j) {
                dotProduct += M[i][i + k - j] * M[i + j][i + k];
            }
            M[i][i + k] = cbrt(dotProduct);
        }

        // Sincronizzazione dei risultati intermedi tra i processi
        for (uint64_t i = 0; i < N - k; ++i) {
            MPI_Bcast(&M[i][i + k], 1, MPI_DOUBLE, i % size, MPI_COMM_WORLD);
        }
    }
}

void printMatrix(const vector<vector<double>>& M, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << setw(10) << fixed << setprecision(7) << M[i][j] << " ";
        }
        cout << endl;
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    uint64_t N = 6;    // default size of the matrix (NxN)
    uint64_t print = 0;

    if (argc != 1 && argc != 2 && argc != 3 ) {
        if(rank == 0) {
            printf("use: %s N p\n", argv[0]);
            printf("     N size of the square matrix\n");   
            printf("     p if 1 print the matrix (optional)\n");  
        }
        MPI_Finalize();
        return -1;
    }
    if (argc > 1) {
        N = stol(argv[1]);
        if(argc > 2){
            print = stol(argv[2]);
        }
    }

    // allocate the matrix
    vector<vector<double>> M(N, vector<double>(N, 0.0));

    auto init=[&]() {
        for(uint64_t i=0; i < N; ++i){
            M[i][i] = static_cast<double>(i + 1) / N;
        }
    };

    init();

    TIMERSTART(wavefront);
    wavefront(M, N, rank, size);
    TIMERSTOP(wavefront);

    if(print == 1 && rank == 0){
        printMatrix(M,N);
    }

    MPI_Finalize();
    return 0;
}
