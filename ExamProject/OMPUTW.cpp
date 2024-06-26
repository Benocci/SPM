#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <omp.h>
#include <hpc_helpers.hpp>

using namespace std;


void wavefront(
    vector<vector<double>>& M,
    const uint64_t &N,
    int numThreads) {

    omp_set_num_threads(numThreads);

	for (uint64_t k = 1; k < N; ++k) {
        #pragma omp parallel for
        for (uint64_t i = 0; i < N - k; ++i) {
            double dotProduct = 0.0;
            for (uint64_t j = 0; j < k + 1; ++j) {
                dotProduct += M[i][i + k - j] * M[i + j][i + k];
            }
            M[i][i + k] = cbrt(dotProduct);
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
	uint64_t N = 6;          // default size of the matrix (NxN)
    uint64_t numThreads = 4; // default number of threads
    uint64_t print = 0;

    if (argc < 1 || argc > 4) {
		printf("use: %s N numThreads p\n", argv[0]);
		printf("     N size of the square matrix\n");	
		printf("     numThread number of thread (optional)\n");
        printf("     p if 1 print the matrix (optional)\n");	
		return -1;
	}
	if (argc > 1) {
		N = stol(argv[1]);
        if(argc > 2){
            numThreads = stol(argv[2]);
            if(argc > 3){
                print = stol(argv[3]);
            }
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
	wavefront(M, N, numThreads);
    TIMERSTOP(wavefront);

    if(print == 1){
        printMatrix(M,N);
    }

    return 0;
}
