# TCL script for running C Simulation with FLOAT version (Phase 1)

# Source directory containing test vectors
set test_dir "../build/test_vectors"

# Open the project (create if doesn't exist)
open_project graphsage_hls_float -reset

# Set top-level function
set_top graphsage_network

# Add source files (FLOAT version)
add_files graphsage_layer_float.cpp -cflags "-I. -DCSIM -std=c++11"
add_files graphsage_layer_float.h -cflags "-I. -DCSIM -std=c++11"

# Add testbench (FLOAT version)
add_files -tb testbench_float.cpp -cflags "-I. -DCSIM -std=c++11"

# Create solution
open_solution "solution1" -flow_target vivado
set_part {xc7z020clg484-1}
create_clock -period 10 -name default

# Run C Simulation
csim_design

# Exit
exit
