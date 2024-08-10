#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <mpi.h>
#include <hpc_helpers.hpp>

using namespace std;

void printMatrix(const vector<vector<double>>& M, uint64_t N) {
    for (uint64_t i = 0; i < N; ++i) {
        for (uint64_t j = 0; j < N; ++j) {
            cout << setw(10) << fixed << setprecision(7) << M[i][j] << " ";
        }
        cout << endl;
    }
}

void print_matrix(double *M, uint64_t N, uint64_t row) {
    for (uint64_t i = 0; i < row; ++i) {
        for (uint64_t j = 0; j < N; ++j) {
            printf("%lf ", M[i * N + j]);
        }
        printf("\n");
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
    for (uint64_t k = 1; k < N; ++k) {  // k = 1 to N-1 (diagonals)
        int currentSize = size;
        if (N - k < (uint64_t) currentSize) {
            currentSize = N - k;
        }

        if (myRank < currentSize) {
            // The computation is divided by rows
            int numberOfRows = (N - k) / currentSize;
            // printf("The number of ROWS per process are %d and rank is %d\n", numberOfRows, myRank);
            int myRows = numberOfRows + k;  // this plus overlap is necessary because to compute the dot product a process needs at least of two rows

            // For the cases that 'rows' is not multiple of size
            if ((uint64_t) myRank < (N - k) % currentSize) {
                myRows++;
            }
            // printf("Rank=%d, k=%d and myRows are %d\n", myRank, k, myRows);

            // The process 0 must specify how many rows are sent to each process   
            vector<int> sendCounts(currentSize);
            vector<int> displs(currentSize);

            if (!myRank) {
                displs[0] = 0;

                for(uint64_t i=0; i < (uint64_t) currentSize; i++){
                    if(i>0){
                        displs[i] = displs[i-1]+sendCounts[i-1]- k * N;
                    }

                    if(i < (N-k) % currentSize){
                        sendCounts[i] = (numberOfRows+k+1)*N;
                    } else {
                        sendCounts[i] = (numberOfRows+k)*N;
                    }

                    // printf("The sendCounts of %d is %d\n", i, sendCounts[i]);
                    // printf("The displs of %d is %d\n", i, displs[i]);
                }
            }

            // Scatter the input matrix
            vector<double> sendBuffer, recvBuffer(myRows * N);
            if (myRank == 0) {
                for (const auto& row : M) {
                    sendBuffer.insert(sendBuffer.end(), row.begin(), row.end());
                }
                
                if(k == 1){
                    //printf("Starting matrix:\n");
                    //printMatrix(M,N);
                    //printf("-------------------\n");
                    //print_matrix(sendBuffer.data(),N);
                }
            }


            MPI_Scatterv(sendBuffer.data(), sendCounts.data(), displs.data(), MPI_DOUBLE, recvBuffer.data(), myRows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            int shift = myRank * numberOfRows;
            if ((N - k) % currentSize != 0 && myRank != 0) {
                shift += (uint64_t) myRank < (N - k) % (uint64_t) currentSize ? myRank : (N - k) % currentSize;
            }
            //printf("MyRank is %d and my shift is %d\n", myRank, shift);
            //print_matrix(recvBuffer.data(), N, myRows);

            // Each process computes its part of the diagonal
            for (uint64_t i = 0; i < myRows - k; ++i) {
                double result = 0.0;

                // #pragma omp parallel for num_threads(numThreads) reduction(+:result)
                for (uint64_t j = 1; j < k + 1; ++j) {
                    //printf("My rank is %d, buffer[%ld]=buffer[%ld]*buffer[%ld]=%f*%f.\n",myRank, shift + i * N + (i + k), shift + i * N + (i + k - j), shift + (i + j) * N + (i + k), recvBuffer[shift + i * N + (i + k - j)], recvBuffer[shift + (i + j) * N + (i + k)]);
                    result += recvBuffer[shift + i * N + (i + k - j)] * recvBuffer[shift + (i + j) * N + (i + k)];
                }
                //printf("My rank is %d, buffer[%ld]=%f\n",myRank, shift+i * N + (i + k),cbrt(result));
                recvBuffer[shift + i * N + (i + k)] = cbrt(result);
            }

            MPI_Gatherv(recvBuffer.data(), myRows * N, MPI_DOUBLE, sendBuffer.data(), sendCounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (myRank == 0) {
                for (uint64_t i = 0; i < N; ++i) {
                    for (uint64_t j = 0; j < N; ++j) {
                        M[i][j] = sendBuffer[i * N + j];
                    }
                }
                //printMatrix(M,N);
            }
        }
    }
    
    double end = MPI_Wtime();

    if(myRank==0){
        std::cout << "Time with " << size << " processes: " << end-start << " seconds" << std::endl;
        if(print == 1){
            printf("PRINT MATRIX:\n");
            //printMatrix(M,N);
        }
        else if(print == 2){
            printf("Last value [0][%ld]=%f\n",N-1, M[0][N-1]);
        }
    }

    MPI_Finalize();
    return 0;
}