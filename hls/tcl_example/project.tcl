#-------------------------------------------------------------
# project.tcl — auto-generated for dt_interface. DO NOT EDIT BY HAND.
#-------------------------------------------------------------

# ---------- core project variables --------------------------
# Absolute path to the HLS project dir for this module
set project_dir   "/home/pelayo/work/NEW_OMTF/omtf_v2/orchestrator/../build_hls/dt_interface"

# Logical metadata (not strictly required by Vitis, but useful)
set project_name  "dt_interface"
set top_function  "dt_omtf_interface"
set part_name     "xcvu13p-fsga2577-1-e"
set clock_period  2.78

set description   "Streaming HLS Module"
set display_name  "dt_interface"
set vendor        "OMTF"
set version       "1.0"

puts "=========================================="
puts "  PROJECT SETUP - dt_interface"
puts "=========================================="
puts "▶ project_dir   = $project_dir"
puts "▶ top_function  = $top_function"
puts "▶ part_name     = $part_name"
puts "▶ clock_period  = $clock_period"
puts ""

# ---------- (re)create project ------------------------------
open_project -reset $project_dir
set_top $top_function

# ---------- solution / timing / device ----------------------
open_solution -reset "solution1" -flow_target vivado
set_part     $part_name
create_clock -period $clock_period -name default

# ---------- sources / testbench -----------------------------
# These filenames are intentionally kept RELATIVE (../../src/...)
# so that Vitis HLS copies them into the out-of-context project
# instead of baking absolute /home/... paths.
set src_files { "/home/pelayo/work/NEW_OMTF/omtf_v2/algo/dt_interface/OMTF_dt_interface.cpp"  }
set tb_files  { "/home/pelayo/work/NEW_OMTF/omtf_v2/algo/dt_interface/OMTF_dt_interface.cpp" "/home/pelayo/work/NEW_OMTF/omtf_v2/verify/tests/tb_dt_interface.cpp" "/home/pelayo/work/NEW_OMTF/omtf_v2/verify/src/xml_parser.cpp" "/home/pelayo/work/NEW_OMTF/omtf_v2/verify/src/xml_model.cpp" "/home/pelayo/work/NEW_OMTF/omtf_v2/verify/src/extractor.cpp" "/home/pelayo/work/NEW_OMTF/omtf_v2/verify/src/logging.cpp" "/home/pelayo/work/NEW_OMTF/omtf_v2/verify/src/latency_shaper.cpp" "/home/pelayo/work/NEW_OMTF/omtf_v2/verify/src/rules.cpp" "/home/pelayo/work/NEW_OMTF/omtf_v2/verify/adapters/csim/mod_dt_interface_adapter.cpp" "/home/pelayo/work/NEW_OMTF/omtf_v2/verify/adapters/cosim/mod_dt_interface_adapter_cosim.cpp"  }

# ---------- include directories -----------------------------
# We build one giant cflags string with all include dirs and any
# extra flags the Python config gave us.
set cflags "-I/home/pelayo/work/NEW_OMTF/omtf_v2/algo/common -I/home/pelayo/work/NEW_OMTF/omtf_v2/verify/include -DHLS_CSIM_BUILD "

puts "▶ Adding source files..."
foreach file $src_files {
    puts "   + $file"
    add_files $file -cflags $cflags
}

puts "▶ Adding testbench files..."
foreach file $tb_files {
    puts "   + $file (TB)"
    add_files $file -cflags $cflags -tb
}

puts ""
puts "✅ project.tcl setup complete for dt_interface"
puts "   Project dir: $project_dir"
puts "   Solution   : solution1"
puts ""

close_project
exit