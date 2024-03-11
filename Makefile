CXX			:= g++
#CXXFLAGS	:= -O3 -g -ffast-math -Wall -Werror -Wpedantic -std=c++20
CXXFLAGS	:= -O0 -g -ffast-math -Wall -Werror -Wpedantic -std=c++20
LDFLAGS		:= -fopenmp
INCLUDE 	:= include
TEST		:= test
BIN			:= bin
BUILD		:= build
SRCFILES	:= $(shell find $(TEST) -iname *.cpp -print)
OBJFILES	:= $(patsubst $(TEST)/%.cpp, $(BUILD)/%.o, $(SRCFILES))
DEPFILES	:= $(patsubst $(TEST)/%.cpp, $(BUILD)/%.d, $(SRCFILES))

# Executable
$(BIN)/test: $(OBJFILES)
	@mkdir -p $(@D)
	$(CXX) $(LDFLAGS) $^ -o $@

# Object Files
$(BUILD)/%.o: $(TEST)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@ -I$(INCLUDE)

# Automatic Dependency Resolution
$(BUILD)/%.d: $(TEST)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $< -MM -MF $@ -I$(INCLUDE)

clean:
	rm -rf $(BUILD)/*
	rm -rf $(BIN)/*

include $(DEPFILES)
