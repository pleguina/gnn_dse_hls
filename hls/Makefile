# Makefile for GraphSAGE HLS Implementation
# For C++ testbench compilation (without Vivado HLS)

CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall
TARGET = testbench
SOURCES = graphsage_layer.cpp testbench.cpp
HEADERS = graphsage_layer.h

# For HLS synthesis, define HLS flags
HLS_FLAGS = -D__SYNTHESIS__

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

test: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
	rm -rf graphsage_hls/

# HLS-specific targets (require Vivado HLS)
.PHONY: hls-csim hls-synth hls-cosim

hls-csim:
	@echo "Running HLS C Simulation..."
	@echo "Note: This requires Vivado HLS to be installed"
	vivado_hls -f run_hls.tcl -tclargs csim

hls-synth:
	@echo "Running HLS Synthesis..."
	@echo "Note: This requires Vivado HLS to be installed"
	vivado_hls -f run_hls.tcl -tclargs synth

hls-cosim:
	@echo "Running HLS C/RTL Cosimulation..."
	@echo "Note: This requires Vivado HLS to be installed"
	vivado_hls -f run_hls.tcl -tclargs cosim

help:
	@echo "GraphSAGE HLS Makefile"
	@echo "======================"
	@echo "Targets:"
	@echo "  all        - Compile C++ testbench"
	@echo "  test       - Compile and run testbench"
	@echo "  clean      - Remove build artifacts"
	@echo "  hls-csim   - Run HLS C simulation (requires Vivado HLS)"
	@echo "  hls-synth  - Run HLS synthesis (requires Vivado HLS)"
	@echo "  hls-cosim  - Run HLS cosimulation (requires Vivado HLS)"
	@echo ""
	@echo "Usage:"
	@echo "  make        # Compile testbench"
	@echo "  make test   # Run testbench"
