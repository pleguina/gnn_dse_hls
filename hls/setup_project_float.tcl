# ==============================================================================
# Project Setup TCL for GraphSAGE HLS - FLOAT Reduced Model
# ==============================================================================
# 
# Configuration:
#   - IN_FEATURES = 16 (after projection)
#   - HIDDEN_FEATURES = 24
#   - OUT_FEATURES = 7 (num_classes)
#   - NUM_NODES = 8 (test subgraph)
#
# This matches the reduced model from model_config.yaml
# ==============================================================================

set project_dir   "../build/hls/graphsage_float"
set project_name  "graphsage_float"
set top_function  "graphsage_network"
set part_name     "xcvu13p-fsga2577-1-e"
set clock_period  10

puts "=========================================="
puts "  PROJECT SETUP - GraphSAGE FLOAT"
puts "=========================================="
puts "▶ Configuration: Reduced Model"
puts "  - IN_FEATURES:     16"
puts "  - HIDDEN_FEATURES: 24"
puts "  - OUT_FEATURES:    7"
puts "  - NUM_NODES:       8"
puts ""
puts "▶ project_dir   = $project_dir"
puts "▶ top_function  = $top_function"
puts "▶ part_name     = $part_name"
puts "▶ clock_period  = $clock_period ns"
puts ""

# ---------- (re)create project ------------------------------
open_project -reset $project_dir
set_top $top_function

# ---------- solution / timing / device ----------------------
open_solution -reset "solution1" -flow_target vivado
set_part     $part_name
create_clock -period $clock_period -name default

# ---------- sources / testbench -----------------------------
# Copy source files to project directory so relative paths work
# (Vitis HLS converts absolute paths to relative in the project file)
set hls_root [file normalize [pwd]]
set src_files { "graphsage_layer_float.cpp" "graphsage_layer_float.h" }
set tb_files  { "testbench_float.cpp" }

# Copy files to project directory
file mkdir $project_dir
puts "▶ Copying source files to project directory..."
foreach file $src_files {
    set src "$hls_root/$file"
    set dst "$project_dir/$file"
    file copy -force $src $dst
    puts "   + Copied $file"
}
foreach file $tb_files {
    set src "$hls_root/$file"
    set dst "$project_dir/$file"
    file copy -force $src $dst
    puts "   + Copied $file (TB)"
}

puts "▶ Adding source files..."
foreach file $src_files {
    puts "   + $file"
    add_files "$file"
}

puts "▶ Adding testbench files..."
foreach file $tb_files {
    puts "   + $file (TB)"
    add_files "$file" -tb
}

# ---------- Configuration -----------------------------------
config_compile -name_max_length 256
config_compile -pipeline_loops 64
config_interface -m_axi_addr64

# ---------- Directives for better performance ---------------
# These can be customized after initial synthesis

puts ""
puts "=========================================="
puts "  Project setup complete!"
puts "=========================================="
puts ""
puts "Next steps:"
puts "  1. Run C simulation:    vitis_hls -f run_csim_float.tcl"
puts "  2. Run synthesis:       vitis_hls -f run_synthesis_float.tcl"
puts "  3. Check reports in:    $project_dir/solution1/syn/report/"
puts ""

close_project
exit
