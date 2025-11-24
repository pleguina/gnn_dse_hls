# ==============================================================================
# Vivado HLS TCL Script for GraphSAGE FPGA Implementation
# ==============================================================================
# This script automates the HLS workflow: C simulation, synthesis, and cosimulation
#
# Usage:
#   vivado_hls -f run_hls.tcl                    # Run all steps
#   vivado_hls -f run_hls.tcl -tclargs csim      # C simulation only
#   vivado_hls -f run_hls.tcl -tclargs synth     # Synthesis only
#   vivado_hls -f run_hls.tcl -tclargs cosim     # Cosimulation only
# ==============================================================================

# Parse command line arguments
set command "all"
if {$argc > 0} {
    set command [lindex $argv 0]
}

puts "===================================================================="
puts "GraphSAGE HLS Automation Script"
puts "Command: $command"
puts "===================================================================="

# ==============================================================================
# Project Configuration
# ==============================================================================

# Project settings
set project_name "graphsage_hls"
set top_function "graphsage_network"
set solution_name "solution1"

# Target device (Xilinx Zynq-7000 series - adjust as needed)
# Common alternatives:
#   xc7z020clg484-1  (Zynq-7000 SoC, smaller)
#   xc7z045ffg900-2  (Zynq-7000 SoC, larger)
#   xczu9eg-ffvb1156-2-e (Zynq UltraScale+)
set target_device "xc7z020clg484-1"

# Clock period in nanoseconds (10ns = 100MHz)
set clock_period 10

# Source files
set source_files {
    "graphsage_layer.cpp"
    "graphsage_layer.h"
}

# Testbench files
set testbench_files {
    "testbench.cpp"
}

# ==============================================================================
# Create Project
# ==============================================================================

# Remove old project if it exists
if {[file exists $project_name]} {
    puts "Removing old project directory..."
    file delete -force $project_name
}

# Create new project
open_project $project_name

# Set top function
set_top $top_function

# Add source files
foreach f $source_files {
    if {[file exists $f]} {
        add_files $f
        puts "Added source file: $f"
    } else {
        puts "WARNING: Source file not found: $f"
    }
}

# Add testbench files
foreach f $testbench_files {
    if {[file exists $f]} {
        add_files -tb $f
        puts "Added testbench file: $f"
    } else {
        puts "WARNING: Testbench file not found: $f"
    }
}

# ==============================================================================
# Create Solution
# ==============================================================================

open_solution $solution_name

# Set target device
set_part $target_device

# Create clock constraint
create_clock -period $clock_period -name default

# Solution configuration
config_compile -name_max_length 256
config_compile -pipeline_loops 64

# Interface configuration
config_interface -m_axi_addr64

# RTL configuration (commented out for Vitis HLS compatibility)
# config_rtl -reset all -reset_async

puts "Project configured:"
puts "  - Top function: $top_function"
puts "  - Target device: $target_device"
puts "  - Clock period: ${clock_period}ns"

# ==============================================================================
# HLS Directives (Optimization Hints)
# ==============================================================================

# Apply optimization directives to the top function
# These can be adjusted based on your performance/resource trade-offs

# Array partitioning for parallel access
set_directive_array_partition -type cyclic -factor 4 -dim 2 "$top_function" input
set_directive_array_partition -type cyclic -factor 4 -dim 2 "$top_function" output
set_directive_array_partition -type cyclic -factor 4 -dim 2 "$top_function" weights1
set_directive_array_partition -type cyclic -factor 4 -dim 2 "$top_function" weights2

# Pipeline the top function for better throughput
set_directive_pipeline "$top_function"

# Interface pragmas - use BRAM for arrays
set_directive_interface -mode bram "$top_function" adj_matrix
set_directive_interface -mode bram "$top_function" input
set_directive_interface -mode bram "$top_function" weights1
set_directive_interface -mode bram "$top_function" bias1
set_directive_interface -mode bram "$top_function" weights2
set_directive_interface -mode bram "$top_function" bias2
set_directive_interface -mode bram "$top_function" output

# Control interface
set_directive_interface -mode ap_ctrl_chain "$top_function"

puts "HLS directives applied for optimization"

# ==============================================================================
# Run HLS Steps
# ==============================================================================

if {$command == "csim" || $command == "all"} {
    puts "\n===================================================================="
    puts "Running C Simulation..."
    puts "===================================================================="
    csim_design -clean

    if {[file exists "csim.log"]} {
        puts "\nC Simulation completed. Check csim.log for details."
    }
}

if {$command == "synth" || $command == "all"} {
    puts "\n===================================================================="
    puts "Running C Synthesis..."
    puts "===================================================================="
    csynth_design

    puts "\nSynthesis completed. Reports available in:"
    puts "  ${project_name}/${solution_name}/syn/report/"
}

if {$command == "cosim" || $command == "all"} {
    puts "\n===================================================================="
    puts "Running C/RTL Cosimulation..."
    puts "===================================================================="

    # Cosimulation options:
    # -rtl verilog|vhdl: RTL language (default: verilog)
    # -trace_level none|all|port: Waveform detail level
    cosim_design -rtl verilog -trace_level none

    puts "\nCosimulation completed. Reports available in:"
    puts "  ${project_name}/${solution_name}/sim/report/"
}

if {$command == "export" || $command == "all"} {
    puts "\n===================================================================="
    puts "Exporting RTL..."
    puts "===================================================================="

    # Export options:
    # -format ip_catalog: Package as Vivado IP
    # -format syn_dcp: Synthesized design checkpoint
    # -format sysgen: System Generator block
    export_design -format ip_catalog -display_name "GraphSAGE Accelerator" -vendor "user" -version "1.0"

    puts "\nRTL export completed. IP available in:"
    puts "  ${project_name}/${solution_name}/impl/"
}

# ==============================================================================
# Summary
# ==============================================================================

puts "\n===================================================================="
puts "HLS Script Execution Complete"
puts "===================================================================="
puts "Project: $project_name"
puts "Solution: $solution_name"
puts "\nKey outputs:"
puts "  - Project directory: ${project_name}/"
puts "  - Synthesis reports: ${project_name}/${solution_name}/syn/report/"
puts "  - Cosim reports: ${project_name}/${solution_name}/sim/report/"
puts "\nTo view results in GUI:"
puts "  vivado_hls -p ${project_name}"
puts "===================================================================="

# Close project
close_project

exit
