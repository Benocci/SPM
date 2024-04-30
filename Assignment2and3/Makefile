CXX                = g++ -std=c++17
CXXFLAGS           = -O3 -march=native -fopenmp
INCLUDES           = -I ./include
SOURCES            = $(wildcard *.cpp)
TARGET             = $(SOURCES:.cpp=)

.PHONY: all clean cleanall Word-Count-seq Word-Count-par

%: %.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTFLAGS) -o $@ $<

Word-Count-seq: Word-Count-seq.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTFLAGS) -o Word-Count-seq Word-Count-seq.cpp

Word-Count-par: Word-Count-par.cpp
	$(CXX) $(INCLUDES) $(CXXFLAGS) $(OPTFLAGS) -o Word-Count-par Word-Count-par.cpp

all: $(TARGET)

clean: 
	-rm -fr *.o *~
cleanall: clean
	-rm -fr $(TARGET)