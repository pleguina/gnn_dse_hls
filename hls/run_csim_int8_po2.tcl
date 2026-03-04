# ==============================================================================
# C Simulation TCL for GraphSAGE INT8-PO2 HLS
# ==============================================================================

set project_dir "../build/hls/graphsage_hls_int8_po2"

puts "=========================================="
puts "  C SIMULATION - GraphSAGE INT8-PO2"
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
    puts "  ❌ C SIMULATION FAILED"
    puts "=========================================="
    puts "Error: $errMsg"
    puts ""
    exit 1
} else {
    puts "=========================================="
    puts "  ✅ C SIMULATION PASSED"
    puts "=========================================="
    puts ""
}

puts "✓ C Simulation complete"
exit 0
