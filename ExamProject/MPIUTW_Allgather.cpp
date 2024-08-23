#include <iostream>
#include <cmath>
#include <vector>
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

    // Temporary buffer to store local results
    vector<double> local_results(N * (N - 1) / 2, 0.0); // Size based on triangular number
    vector<int> counts(size, 0); // Count of elements each process contributes
    vector<int> displs(size, 0); // Displacements for each process in the final gathered array

    // Calculate counts and displacements for MPI_Allgatherv
    int offset = 0;
    for (int p = 0; p < size; ++p) {
        counts[p] = 0;
        for (uint64_t k = 1; k < N; ++k) {
            for (uint64_t i = p; i < N - k; i += size) {
                counts[p]++;
            }
        }
        displs[p] = offset;
        offset += counts[p];
    }

    // Parallel wavefront computation
    offset = 0;
    for (uint64_t k = 1; k < N; ++k) {
        int local_count = 0;
        for (uint64_t i = rank; i < N - k; i += size) {
            double dotProduct = 0.0;
            for (uint64_t j = 1; j < k + 1; ++j) {
                dotProduct += M[i][i+k - j] * M[i+k][i+j];
            }
            M[i][i+k] = cbrt(dotProduct);
            M[i+k][i] = M[i][i+k];
            local_results[offset + local_count] = M[i][i+k];
            local_count++;
        }
        offset += local_count;

        // Synchronize all processes by gathering the results
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, local_results.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // Update the matrix with the gathered results
        offset = 0;
        for (uint64_t k = 1; k < N; ++k) {
            for (uint64_t i = 0; i < N - k; ++i) {
                M[i][i+k] = local_results[offset];
                M[i+k][i] = M[i][i+k];
                offset++;
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
