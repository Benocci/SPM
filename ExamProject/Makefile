# 
# FF_ROOT     points to the FastFlow root directory (i.e.
#             the one containing the ff directory).
ifndef FF_ROOT 
FF_ROOT			= ${HOME}/fastflow
endif

CXX             = g++ -std=c++20
CXXFLAGS       += -Wall -march=native -fopenmp
MPI    			= mpicxx -std=c++20
OPTFLAGS	    = -O3 -DNDEBUG -ffast-math
INCLUDES	   	= -I. -I ../include -I $(FF_ROOT)
LIBS            = -pthread
SOURCES         = $(wildcard *.cpp)
TARGET          = $(SOURCES:.cpp=)

.PHONY: all clean cleanall run_single run_average

%: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTFLAGS) -o $@.o $< $(LIBS)

mpi: MPIUTW.cpp
	$(MPI) $(INCLUDES) $(CXXFLAGS) $(OPTFLAGS) -o MPIUTW.o $< $(LIBS)

all: $(TARGET)

run_single:
	./$(FILE) $(ARGS)

# Run the program and calculate the average execution time over 10 runs
run_average: 
	@rm -f temp_elapsed_times.txt; \
	for i in $$(seq 1 10); do \
		./$(FILE) $(ARGS) | grep "# elapsed time" | awk '{print "Time: " $$5}' >> temp_elapsed_times.txt; \
	done; \
	echo "Average elapsed times over 10 executions of $(FILE): $$(grep "Time: " temp_elapsed_times.txt | awk '{ total += $$2; count++ } END { print total/count "s" }')"; \
	rm -f temp_elapsed_times.txt; 

clean: 
	-rm -fr *.o *~
cleanall: clean
	-rm -fr $(TARGET)