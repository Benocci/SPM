//
// Sequential code of the first SPM Assignment a.a. 23/24.
//
// compile:
// g++ -std=c++20 -O3 -march=native -I<path-to-include> UTWavefront.cpp -o UTW
//
#include <barrier>
#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <cassert>
#include <cstdint>
#include <mutex>
#include <hpc_helpers.hpp>

int random(const int &min, const int &max) {
	static std::mt19937 generator(117);
	std::uniform_int_distribution<int> distribution(min,max);
	return distribution(generator);
};		

// emulate some work
void work(std::chrono::microseconds w) {
	auto end = std::chrono::steady_clock::now() + w;
    while(std::chrono::steady_clock::now() < end);	
}

void wavefront_sequential(
    const std::vector<int> &M,
    const uint64_t &N) {

	for(uint64_t k = 0; k< N; ++k) {        // for each upper diagonal
		for(uint64_t i = 0; i< (N-k); ++i) {// for each elem. in the diagonal
            work(std::chrono::microseconds(M[i*N+(i+k)]));
		}
	}
}

void wavefront_parallelB(
    const std::vector<int> &M,
    const uint64_t &N,
	const uint64_t &num_threads) {

	std::barrier my_barrier(num_threads);

    const uint64_t chunk_size = N/num_threads;

	auto static_parallelization = [&] (const uint64_t& id) -> void {
		uint64_t start = id * chunk_size;
        uint64_t end = (id == num_threads - 1) ? N : start + chunk_size;

		for(uint64_t k = 0; k<N; ++k) { // for each upper diagonal
            for(uint64_t i = start; i < end && i < N - k; ++i) {// for each elem. in the diagonal
                work(std::chrono::microseconds(M[i*N+(i+k)]));
			}

			if(id+1 > N-k){
				my_barrier.arrive_and_drop();
				break;
			}
			else{
				my_barrier.arrive_and_wait();
			}
		}

    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (uint64_t i = 0; i < num_threads; i++){
        threads.emplace_back(static_parallelization, i);
    }

    for (auto& thread : threads)
        thread.join();
}

void wavefront_parallel(
    const std::vector<int> &M,
    const uint64_t &N,
	const uint64_t &num_threads) {

	std::barrier my_barrier(num_threads);

	auto static_parallelization = [&] (const uint64_t& id) -> void {
		const uint64_t off = id;
		const uint64_t str = num_threads;


		for(uint64_t k = 0; k<N; ++k) {

			for(u_int64_t i = off; i<N-k; i+=str){
				work(std::chrono::microseconds(M[i*N+(i+k)]));
			}

			if(id+1 > N-k){
				my_barrier.arrive_and_drop();
				break;
			}
			else{
				my_barrier.arrive_and_wait();
			}
		}

    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (uint64_t i = 0; i < num_threads; i++){
        threads.emplace_back(static_parallelization, i);
    }

    for (auto& thread : threads)
        thread.join();
}

void wavefront_dynamic(
    const std::vector<int> &M,
    const uint64_t &N,
	const uint64_t &num_threads){

	volatile std::atomic<uint64_t> number_threads {num_threads};
	volatile std::atomic<uint64_t> row {0};
	uint64_t k = 0;

	auto on_completition = [&]() {
		k++;
		row.store(0);
	};

	std::barrier my_barrier(num_threads, on_completition);

    auto dynamic_paralizzation = [&] (const uint64_t& id) -> void {
		uint64_t _row;
		uint64_t _number_threads;

		while(k<N) {
			
			_row = row.fetch_add(1);

			if(_row < N-k){
				work(std::chrono::microseconds(M[_row * N + (_row + k)]));
			}
			else{
				if(id+1 > N-k){
					my_barrier.arrive_and_drop();
					break;
				}
				else{
					my_barrier.arrive_and_wait();
				}
			}
		}

    };

    
    std::vector<std::thread> threads;
	threads.reserve(num_threads);
	std::atomic<uint64_t> atomic_row {0};

    for (uint64_t id = 0; id < num_threads; id++)
        threads.emplace_back(dynamic_paralizzation, id);

    for (auto& thread : threads)
        thread.join();

}


int main(int argc, char *argv[]) {
	int min    = 0;      // default minimum time (in microseconds)
	int max    = 1000;   // default maximum time (in microseconds)
	uint64_t N = 6;    // default size of the matrix (NxN)
	uint64_t num_threads = 6; 
	
	if (argc != 1 && argc != 2 && argc != 5) {
		std::printf("use: %s N [min max]\n", argv[0]);
		std::printf("     N size of the square matrix\n");
		std::printf("     min waiting time (us)\n");
		std::printf("     max waiting time (us)\n");	
		std::printf("     num_threads number of threads to use (optional)\n");	
		return -1;
	}
	if (argc > 1) {
		N = std::stol(argv[1]);
		if (argc > 2) {
			min = std::stol(argv[2]);
			max = std::stol(argv[3]);
		}
		if (argc > 4){
			num_threads = std::stol(argv[4]);
		}
	}

	if(num_threads > N){
		std::printf("Too much thread, number reduced to %d\n", N);
		num_threads = N;
	}

	// allocate the matrix
	std::vector<int> M(N*N, -1);

	uint64_t expected_totaltime=0;
	// init function
	auto init=[&]() {
		for(uint64_t k = 0; k< N; ++k) {  
			for(uint64_t i = 0; i< (N-k); ++i) {  
				int t = random(min,max);
				M[i*N+(i+k)] = t;
				expected_totaltime +=t;				
			}
		}
	};
	
	init();

	std::printf("Sequential code stimated compute time ~ %f (ms)\n", expected_totaltime/1000.0);
	std::printf("Parallel dynamic code executed (N = %d) with %d thread\n",N , num_threads);

    //std::printf("Sequential code stimated (N = %d) compute time ~ %f (ms)\n", N, expected_totaltime/1000.0);
    TIMERSTART(wavefront_sequential);
	wavefront_sequential(M, N);
    TIMERSTOP(wavefront_sequential);

	//std::printf("Parallel code executed (N = %d) with %d thread\n",N , num_threads);
	TIMERSTART(wavefront_parallel_static);
	wavefront_parallel(M, N, num_threads); 
    TIMERSTOP(wavefront_parallel_static);

	// SECOND VERSION (NOT OPTIMAL)
	//TIMERSTART(wavefront_parallelB);
	//wavefront_parallelB(M, N, num_threads); 
    //TIMERSTOP(wavefront_parallelB);

	//std::printf("Parallel dynamic code executed (N = %d) with %d thread\n",N , num_threads);
	TIMERSTART(wavefront_parallel_dynamic);
	wavefront_dynamic(M, N, num_threads); 
    TIMERSTOP(wavefront_parallel_dynamic);

    return 0;
}
