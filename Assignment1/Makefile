CXX                = g++ -std=c++20
CXXFLAGS	   = -O3 -march=native
INCLUDES	   = -I ./include
SOURCES            = UTW.cpp
EXECUTABLE = UTW
OBJ = $(SRC:.cpp=.o)

# Target principale
all: $(EXECUTABLE)

# g++ -std=c++20 -O3 -march=native -I<path-to-include> UTWavefront.cpp -o UTW


.PHONY: all clean cleanall 

# Regola per la compilazione del programma eseguibile
$(EXECUTABLE): $(OBJ)
	$(CXX) $(LDFLAGS) $^ -o $@

# Regola per la compilazione dei file sorgente
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Pulisce i file oggetto e l'eseguibile
clean:
	rm -f $(OBJ) $(EXECUTABLE)

# Esegue una singola esecuzione del programma
run_single:
	@./$(EXECUTABLE) $(ARGS)

# Esegue il programma e calcola la media dei tempi di esecuzione su 10 esecuzioni
run_average:
	@rm -f temp_elapsed_times.txt; \
	for i in $$(seq 1 10); do \
		./$(EXECUTABLE) $(ARGS) | grep "# elapsed time" | awk '{print $$4 " " $$5}' >> temp_elapsed_times.txt; \
	done; \
	echo "Average elapsed times over 10 executions:"; \
	echo "Sequential code: $$(grep "sequential" temp_elapsed_times.txt | awk '{ total += $$2; count++ } END { print total/count "s" }')"; \
	echo "Parallel element cyclic code: $$(grep "element" temp_elapsed_times.txt | awk '{ total += $$2; count++ } END { print total/count "s" }')"; \
	echo "Parallel dynamic code: $$(grep "dynamic" temp_elapsed_times.txt | awk '{ total += $$2; count++ } END { print total/count "s" }')"; \
	rm -f temp_elapsed_times.txt; 

# Phony target per i comandi non legati a file
.PHONY: all clean run run_average