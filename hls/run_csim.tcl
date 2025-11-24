# ==============================================================================
# C Simulation TCL for GraphSAGE HLS
# ==============================================================================

set project_dir "../build/hls/graphsage_hls"

puts "=========================================="
puts "  C SIMULATION - GraphSAGE"
puts "=========================================="
puts ""

puts "▶ Opening project: $project_dir"
open_project $project_dir

puts "▶ Opening solution: solution1"
open_solution "solution1"

set csim_opts "-clean"

puts ""
puts "=========================================="
puts "  STARTING C SIMULATION"
puts "=========================================="

puts "▶ Command: csim_design $csim_opts"
set status [catch {eval csim_design $csim_opts} errMsg]
puts ""

if {$status != 0} {
    puts "=========================================="
    puts "  ❌ C-SIMULATION FAILED"
    puts "=========================================="
    puts "Error: $errMsg"
    exit 1
} else {
    puts "=========================================="
    puts "  ✅ C-SIMULATION PASS"
    puts "=========================================="
}

close_project
exit
