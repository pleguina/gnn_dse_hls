# TCL script for C simulation of GraphSAGE INT8 Quantized (PTQ) implementation
# Run with: vitis_hls -f run_csim_qat.tcl

# Create new project
open_project -reset graphsage_qat
set_top graphsage_network_qat

# Add source files
add_files graphsage_layer_qat.cpp -cflags "-I."
add_files graphsage_layer_qat.h -cflags "-I."

# Add testbench
add_files -tb testbench_qat.cpp -cflags "-I."

# Set solution
open_solution "solution1" -flow_target vivado
set_part {xcvu13p-fsga2577-1-e}
create_clock -period 2.77 -name default

# Run C simulation
csim_design

exit
