#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <mpi.h>

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
    uint64_t N = 6;    // default size of the matrix (NxN)
    uint64_t print = 0;

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 1 && argc != 2 && argc != 3 ) {
        printf("use: %s N p\n", argv[0]);
        printf("     N size of the square matrix\n");
        printf("     p if 1 print the matrix, if 2 print last value (optional)\n");
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

    // Measure the current time
    double start = MPI_Wtime();

    vector<int> sendcounts(size, 0);
    vector<int> displs(size, 0);

    for (uint64_t k = 1; k < N; ++k) {
        // Determine the number of elements each process will handle
        int idx = 0;
        for (int i = 0; i < size; ++i) {
            sendcounts[i] = 0;
            for (uint64_t j = i; j < N - k; j += size) {
                sendcounts[i]++;
            }
        }

        // Calculate displacements
        displs[0] = 0;
        for (int i = 1; i < size; ++i) {
            displs[i] = displs[i-1] + sendcounts[i-1];
        }

        vector<double> local_data(sendcounts[rank], 0.0);
        vector<double> gathered_data((N * (N - 1)) / 2, 0.0);

        idx = 0;
        for (uint64_t i = rank; i < N - k; i += size) {
            double dotProduct = 0.0;
            for (uint64_t j = 1; j < k + 1; ++j) {
                dotProduct += M[i][i+k - j] * M[i+k][i+j];
            }
            M[i][i+k] = cbrt(dotProduct);
            local_data[idx++] = M[i][i+k];
        }

        // Allgatherv to gather data from all processes
        MPI_Allgatherv(local_data.data(), sendcounts[rank], MPI_DOUBLE,
                       gathered_data.data(), sendcounts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        idx = 0;
        for (uint64_t i = 0; i < N - k; ++i) {
            if (i % size == rank) {
                M[i][i+k] = gathered_data[displs[rank] + idx];
                M[i+k][i] = M[i][i+k];
                idx++;
            } else {
                M[i][i+k] = gathered_data[displs[i % size] + (i / size)];
                M[i+k][i] = M[i][i+k];
            }
        }
    }

    double end = MPI_Wtime();

    if (rank == 0) {
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
