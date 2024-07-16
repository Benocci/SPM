#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <mpi.h>
#include <hpc_helpers.hpp>

using namespace std;

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
    
    int myRank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    uint64_t N = 6;    // default size of the matrix (NxN)
    uint64_t print = 0;

    if (argc != 1 && argc != 2 && argc != 3 ) {
        if(myRank == 0) {
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

    if(myRank==0){
        init();
    }

    // Measure the current time
	double start = MPI_Wtime();

    // Distribute work across processes
    for (int k = 1; k < N; ++k) {  // k = 1 to N-1 (diagonals)
        if (N - k < size) {
            size--;
        }

        if (myRank < size) {
            // The computation is divided by rows
            int overlap = 1;  // number of overlapping rows
            int numberOfRows = (N - k) / size;
            // printf("The number of ROWS per process are %d and rank is %d\n", numberOfRows, myRank);
            int myRows = numberOfRows + k;  // this plus overlap is necessary because to compute the dot product a process needs at least of two rows

            // For the cases that 'rows' is not multiple of size
            if (myRank < (N - k) % size) {
                myRows++;
            }
            // printf("Rank=%d, k=%d and myRows are %d\n", myRank, k, myRows);

            // Arrays for the chunk of data to work
            vector<vector<double>> myData(myRows, vector<double>(N, 0.0));

            // The process 0 must specify how many rows are sent to each process   
            vector<int> sendCounts(size);
            vector<int> displs(size);

            if (!myRank) {
                int currentDisp = 0;

                for (int i = 0; i < size; i++) {
                    if (i < (N - k) % size) {
                        sendCounts[i] = (numberOfRows + k + 1) * N;
                    } else {
                        sendCounts[i] = (numberOfRows + k) * N;
                    }
                    displs[i] = currentDisp;
                    currentDisp += sendCounts[i];
                }
            }

            // Scatter the input matrix
            vector<double> sendBuffer, recvBuffer(myRows * N);
            if (myRank == 0) {
                for (const auto& row : M) {
                    sendBuffer.insert(sendBuffer.end(), row.begin(), row.end());
                }
            }

            MPI_Scatterv(sendBuffer.data(), sendCounts.data(), displs.data(), MPI_DOUBLE, recvBuffer.data(), myRows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            int shift = myRank * numberOfRows;
            if ((N - k) % size != 0 && myRank != 0) {
                shift += myRank < (N - k) % size ? myRank : (N - k) % size;
            }
            // printf("MyRank is %d and my shift is %d\n", myRank, shift);

            // Each process computes its part of the diagonal
            for (int i = 0; i < myRows - k; ++i) {
                double result = 0.0;

                // #pragma omp parallel for num_threads(numThreads) reduction(+:result)
                for (int j = 1; j < k + 1; ++j) {
                    result += recvBuffer[shift + i * N + (i + k - j)] * recvBuffer[shift + (i + j) * N + (i + k)];
                }
                recvBuffer[shift + i * N + (i + k)] = cbrt(result);
            }

            MPI_Gatherv(recvBuffer.data(), myRows * N, MPI_DOUBLE, sendBuffer.data(), sendCounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (myRank == 0) {
                for (int i = 0; i < N; ++i) {
                    for (int j = 0; j < N; ++j) {
                        M[i][j] = sendBuffer[i * N + j];
                    }
                }
            }
        }
    }
    
    double end = MPI_Wtime();

    if(myRank==0){
        std::cout << "Time with " << size << " processes: " << end-start << " seconds" << std::endl;
        if(print == 1){
            printMatrix(M,N);
        }
        else if(print == 2){
            printf("Last value [0][%ld]=%f\n",N-1, M[0][N-1]);
        }
    }

    MPI_Finalize();
    return 0;
}
