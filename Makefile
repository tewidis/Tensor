CXX         := g++
TEST        := test
BIN         := bin
BUILD       := build
LIB         := lib
INCLUDE     := include
BLAS        := $(LIB)/OpenBLAS-0.3.26
FFTW        := $(LIB)/fftw-3.3.10
LDFLAGS     := -fopenmp
LDFLAGS     += -L$(BLAS) -lopenblas -Wl,-rpath $(BLAS)
LDFLAGS     += -L$(FFTW)/.libs -lfftw3f -Wl,-rpath $(FFTW)
HEADERS     := $(addprefix -I, $(INCLUDE) $(BLAS) $(FFTW)/api)
SRCFILES	:= $(shell find $(TEST) -iname *.cpp -print)
OBJFILES	:= $(patsubst $(TEST)/%.cpp, $(BUILD)/%.o, $(SRCFILES))
DEPFILES	:= $(patsubst $(TEST)/%.cpp, $(BUILD)/%.d, $(SRCFILES))
BUILD_TYPE 	:= Debug

ifeq ($(BUILD_TYPE), Debug)
	CXXFLAGS	:= -O0 -g -ffast-math -Wall -Werror -Wpedantic -std=c++20 -DDEBUG
else
	CXXFLAGS	:= -O3 -g -ffast-math -Wall -Werror -Wpedantic -std=c++20 -DNDEBUG
endif

# Executable
$(BIN)/test: $(OBJFILES)
	@mkdir -p $(@D)
	$(CXX) $^ -o $@ $(LDFLAGS)

# Object Files
$(BUILD)/%.o: $(TEST)/%.cpp $(INCLUDE)/*
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(HEADERS)

# Automatic Dependency Resolution
$(BUILD)/%.d: $(TEST)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $< -MM -MF $@ $(HEADERS)

clean:
	rm -rf $(BUILD)/*
	rm -rf $(BIN)/*

include $(DEPFILES)
