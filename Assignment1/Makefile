CXX                = g++ -std=c++20
CXXFLAGS           = -O3 -march=native
INCLUDES           = -I ./include
EXECUTABLE         = UTW
SOURCES            = UTW.cpp
TARGET             = $(SOURCES:.cpp=)

# Main target
all: $(TARGET)

# Rule for compiling source files
%: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) -o $@ $<

# Clean up object files and the executable
clean:
	rm -f $(OBJ) $(EXECUTABLE)

# Run the program once
run_single:
	@./$(EXECUTABLE) $(ARGS)

# Run the program and calculate the average execution time over 10 runs
run_average:
	@rm -f temp_elapsed_times.txt; \
	for i in $$(seq 1 10); do \
		./$(EXECUTABLE) $(ARGS) | grep "# elapsed time" | awk '{print $$4 " " $$5}' >> temp_elapsed_times.txt; \
	done; \
	./$(EXECUTABLE) $(ARGS) | grep "stimated" >> temp_elapsed_times.txt; \
	echo "$$(grep "stimated" temp_elapsed_times.txt)"; \
	echo "Average elapsed times over 10 executions:"; \
	echo "Sequential code: $$(grep "sequential" temp_elapsed_times.txt | awk '{ total += $$2; count++ } END { print total/count "s" }')"; \
	echo "Parallel element cyclic code: $$(grep "element" temp_elapsed_times.txt | awk '{ total += $$2; count++ } END { print total/count "s" }')"; \
	echo "Parallel dynamic code: $$(grep "dynamic" temp_elapsed_times.txt | awk '{ total += $$2; count++ } END { print total/count "s" }')"; \
	rm -f temp_elapsed_times.txt; 

# Phony target for commands not related to files
.PHONY: all clean run_single run_average
