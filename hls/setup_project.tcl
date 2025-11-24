# ==============================================================================
# Project Setup TCL for GraphSAGE HLS
# ==============================================================================

set project_dir   "../build/hls/graphsage_hls"
set project_name  "graphsage_hls"
set top_function  "graphsage_network"
set part_name     "xc7z020clg484-1"
set clock_period  10

puts "=========================================="
puts "  PROJECT SETUP - GraphSAGE"
puts "=========================================="
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
set src_files { "graphsage_layer.cpp" "graphsage_layer.h" }
set tb_files  { "testbench.cpp" }

puts "▶ Adding source files..."
foreach file $src_files {
    puts "   + $file"
    add_files $file
}

puts "▶ Adding testbench files..."
foreach file $tb_files {
    puts "   + $file (TB)"
    add_files $file -tb
}

# ---------- Configuration -----------------------------------
config_compile -name_max_length 256
config_compile -pipeline_loops 64
config_interface -m_axi_addr64

puts ""
puts "✅ Project setup complete for GraphSAGE"
puts "   Project dir: $project_dir"
puts "   Solution   : solution1"
puts ""

close_project
exit
