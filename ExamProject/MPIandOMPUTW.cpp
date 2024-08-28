#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <mpi.h>
#include <omp.h>

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

    // Allocate the matrix
    vector<vector<double>> M(N, vector<double>(N, 0.0));

    auto init = [&]() {
        for (uint64_t i = 0; i < N; ++i) {
            M[i][i] = static_cast<double>(i + 1) / N;
        }
    };

    init();

    // Measure the current time
    double start = MPI_Wtime();

    for (uint64_t k = 1; k < N; ++k) {
        vector<double> values_to_send; // Array to collect calculated values
        vector<int> indices;           // Array to store indices corresponding to the calculated values

        #pragma omp parallel
        {
            vector<double> local_values_to_send;
            vector<int> local_indices;

            #pragma omp for schedule(dynamic)
            for (uint64_t i = rank; i < N - k; i += size) {
                double dotProduct = 0.0;
                for (uint64_t j = 1; j < k + 1; ++j) {
                    dotProduct += M[i][i+k - j] * M[i+k][i+j];
                }
                double value = cbrt(dotProduct);
                M[i][i+k] = value;
                M[i+k][i] = value;

                // Store the calculated value and its index
                #pragma omp critical
                {
                    values_to_send.push_back(value);
                    indices.push_back(i);
                }
            }
        }

        // Gather all the calculated values from all processes
        int total_values = values_to_send.size();
        vector<int> recv_counts(size);
        MPI_Allgather(&total_values, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        vector<int> displs(size, 0);
        int total_elements = 0;
        for (int i = 0; i < size; ++i) {
            displs[i] = total_elements;
            total_elements += recv_counts[i];
        }

        vector<double> gathered_values(total_elements);
        vector<int> gathered_indices(total_elements);

        MPI_Allgatherv(values_to_send.data(), total_values, MPI_DOUBLE,
                      gathered_values.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
                      MPI_COMM_WORLD);

        MPI_Allgatherv(indices.data(), total_values, MPI_INT,
                      gathered_indices.data(), recv_counts.data(), displs.data(), MPI_INT,
                      MPI_COMM_WORLD);

        // Update the matrix with the gathered values
        for (int i = 0; i < total_elements; ++i) {
            M[gathered_indices[i]][gathered_indices[i] + k] = gathered_values[i];
            M[gathered_indices[i] + k][gathered_indices[i]] = gathered_values[i];
        }
    }

    double end = MPI_Wtime();

    if (rank == 0) {
        std::cout << "Time with " << size << " processes: " << end - start << " seconds" << std::endl;
        if (print == 1) {
            printMatrix(M, N);
        } else if (print == 2) {
            printf("Last value [0][%ld]=%f\n", N - 1, M[0][N - 1]);
        }
    }

    MPI_Finalize();
    return 0;
}
