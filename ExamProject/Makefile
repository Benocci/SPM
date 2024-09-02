# FF_ROOT points to the FastFlow root directory (i.e., the one containing the ff directory).
# If FF_ROOT is not defined, set it to a default location within the HOME directory.
ifndef FF_ROOT 
FF_ROOT			= ${HOME}/SPM/fastflow
endif

# Compiler settings
CXX             = g++ -std=c++20                 # C++ compiler with C++20 standard
CXXFLAGS       += -Wall -march=native -fopenmp   # Compiler flags: Wall (all warnings), march=native (optimize for the host machine), and OpenMP support
MPI    			= mpicxx -std=c++20              # MPI compiler with C++20 standard
OPTFLAGS	    = -O3 -DNDEBUG -ffast-math        # Optimization flags: O3 (high optimization), DNDEBUG (disable debugging), ffast-math (fast floating-point optimizations)
INCLUDES	   	= -I. -I ../include -I $(FF_ROOT) # Include directories: current directory, a relative path, and the FastFlow root directory
LIBS            = -pthread                       # Libraries to link against: pthread for threading support

# Source file lists
MPIFILES 		= $(wildcard MPI*.cpp)           # All source files starting with "MPI"
SOURCES         = $(filter-out $(MPIFILES), $(wildcard *.cpp)) # All source files except those starting with "MPI"
TARGET          = $(SOURCES:.cpp=)               # Target executables are named after the source files without the .cpp extension

# Phony targets are not actual files and will always be run when called
.PHONY: all clean cleanall run_single run_average run_average_mpi

# Default rule to compile any .cpp file into an executable
%: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTFLAGS) -o $@.o $< $(LIBS)

# Rule to compile MPI source files
mpi: $(MPIFILES:.cpp=.o)

# Rule to compile a single .cpp file into an object file for MPI
%.o: %.cpp
	$(MPI) $(INCLUDES) $(CXXFLAGS) $(OPTFLAGS) -o $@ $< $(LIBS)

# Target to build all non-MPI executables
all: $(TARGET)

# Target to run a single execution of the program
run_single:
	./$(FILE) $(ARGS)

# Target to run the program multiple times and calculate the average execution time over 10 runs
run_average:
	@ ARG2=$(shell echo $(ARGS) | awk '{print $$2}'); \ # Extract the second argument (usually matrix size or a key parameter)
	for i in $$(seq 1 10); do \                         # Loop to run the program 10 times
		./$(FILE) $(ARGS) | grep "# elapsed time" | awk '{print "Time: " $$5}' >> temp_elapsed_times_$$ARG2.txt; \ # Extract elapsed time and store it in a temp file
  	done; \
  	echo "Average elapsed times over 10 executions:"; \ # Print the average of the recorded times
  	echo "$$(grep "Time: " temp_elapsed_times_$$ARG2.txt | awk '{ total += $$2; count++ } END { print total/count "s" }')"; \
  	rm -f temp_elapsed_times_$$ARG2.txt;                # Clean up the temp file after use

# Similar to run_average, but uses MPI for parallel execution
run_average_mpi:
	@ ARG2=$(shell echo $(ARGS) | awk '{print $$2}'); \ # Extract the second argument
	for i in $$(seq 1 10); do \                         # Loop to run the program 10 times using MPI
		mpirun ./$(FILE) $(ARGS) | grep "# elapsed time" | awk '{print "Time: " $$5}' >> temp_elapsed_times_$$ARG2.txt; \ # Extract elapsed time with MPI
  	done; \
  	echo "Average elapsed times over 10 executions:"; \ # Print the average of the recorded times
  	echo "$$(grep "Time: " temp_elapsed_times_$$ARG2.txt | awk '{ total += $$2; count++ } END { print total/count "s" }')"; \
  	rm -f temp_elapsed_times_$$ARG2.txt;                # Clean up the temp file after use

# Target to clean up object files and temporary files
clean: 
	-rm -fr *.o *~

# Target to clean up all generated files, including executables
cleanall: clean
	-rm -fr $(TARGET)
