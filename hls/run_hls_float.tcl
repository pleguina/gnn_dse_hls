# ==============================================================================
# Run HLS Synthesis for GraphSAGE FLOAT Reduced Model
# ==============================================================================

# Source the project setup
source setup_project_float.tcl

puts ""
puts "=========================================="
puts "  Running C Synthesis (FLOAT)"
puts "=========================================="
puts ""

# Run C synthesis
csynth_design

puts ""
puts "=========================================="
puts "  Synthesis Complete"
puts "=========================================="
puts ""
puts "Check synthesis report:"
puts "  ../build/hls/graphsage_float/solution1/syn/report/graphsage_network_csynth.rpt"
puts ""

exit
