#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <ff/parallel_for.hpp>
#include <hpc_helpers.hpp>

#define MAX_THREAD 40
using namespace std;
using namespace ff;

void wavefront(
    vector<vector<double>>& M,
    const uint64_t &N,
    int numThreads) {
    ParallelFor pf(numThreads);

	for (uint64_t k = 1; k < N; ++k) {
        pf.parallel_for(0, N-k, 1, [&, N, k](const int i) {
            double dotProduct = 0.0;
            for (uint64_t j = 0; j < k + 1; ++j) {
                dotProduct += M[i][i + k - j] * M[i + j][i + k];
            }
            M[i][i + k] = cbrt(dotProduct);
        });
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

	// TIMERSTART(wavefront);
	// wavefront(M, N, numThreads);
    // TIMERSTOP(wavefront);

    ffTime(START_TIME);
	wavefront(M, N, numThreads);
    ffTime(STOP_TIME);

    std::cout << "# elapsed time (wavefront): " << ffTime(GET_TIME)/1000  << "s" << std::endl;

    if(print == 1){
        printMatrix(M,N);
    }
    else if(print == 2){
        printf("Last value [0][%ld]=%f\n",N-1, M[0][N-1]);
    }

    return 0;
}
