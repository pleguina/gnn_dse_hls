#!/usr/bin/env python3
"""
Generate Vitis HLS TCL scripts for Fixed-Point GraphSAGE project.
Creates project setup, C simulation, and synthesis scripts.
"""

import os
from jinja2 import Template

# Configuration
config = {
    "module_name": "graphsage_fixed",
    "top": "graphsage_network_fixed",
    "clock_period": 2.77,  # ns (361 MHz - same as float)
    "device": "xcvu13p-fsga2577-1-e",
    "src": ["graphsage_layer_fixed.cpp", "graphsage_layer_fixed.h"],  # Need .cpp for synthesis
    "tb": ["testbench_fixed.cpp"],
    "project_dir": "/home/pelayo/work/simple-gnn/build/hls/graphsage_fixed",
    "hls_dir": "/home/pelayo/work/simple-gnn/hls",
    
    # Fixed-point bit widths (configurable)
    "data_w": 16,
    "data_i": 8,
    "weight_w": 16,
    "weight_i": 4,
    "acc_w": 32,
    "acc_i": 16,
    "scale_w": 16,
    "scale_i": 2,
}

# Project TCL template
project_tcl = Template("""
# Vitis HLS Project Setup - {{ module_name }}
# Fixed-Point GraphSAGE Implementation

# Create and open project
open_project {{ module_name }}

# Add source files
{% for src_file in src %}
add_files {{ hls_dir }}/{{ src_file }} -cflags "-std=c++11 -DDATA_W={{ data_w }} -DDATA_I={{ data_i }} -DWEIGHT_W={{ weight_w }} -DWEIGHT_I={{ weight_i }} -DACC_W={{ acc_w }} -DACC_I={{ acc_i }} -DSCALE_W={{ scale_w }} -DSCALE_I={{ scale_i }}"
{% endfor %}

# Add testbench files
{% for tb_file in tb %}
add_files -tb {{ hls_dir }}/{{ tb_file }} -cflags "-std=c++11 -DDATA_W={{ data_w }} -DDATA_I={{ data_i }} -DWEIGHT_W={{ weight_w }} -DWEIGHT_I={{ weight_i }} -DACC_W={{ acc_w }} -DACC_I={{ acc_i }} -DSCALE_W={{ scale_w }} -DSCALE_I={{ scale_i }}"
{% endfor %}

# Set top function
set_top {{ top }}

# Create solution
open_solution "solution1" -flow_target vivado

# Set target device
set_part {{ device }}

# Set clock period
config_schedule -enable_dsp_full_reg=false
create_clock -period {{ clock_period }} -name default

exit
""")

# C Simulation TCL template
csim_tcl = Template("""
# Vitis HLS C Simulation - {{ module_name }}

open_project {{ module_name }}
open_solution "solution1"

# Run C simulation
csim_design -clean

close_project
exit
""")

# Synthesis TCL template
synth_tcl = Template("""
# Vitis HLS Synthesis - {{ module_name }}

open_project {{ module_name }}
open_solution "solution1"

# Run C synthesis
csynth_design

# Export design (optional)
# export_design -format ip_catalog

close_project
exit
""")

def generate_tcl_scripts():
    """Generate all TCL scripts."""
    
    # Create output directory
    os.makedirs(config["project_dir"], exist_ok=True)
    
    # Generate project.tcl
    with open(f"{config['project_dir']}/project.tcl", "w") as f:
        f.write(project_tcl.render(**config))
    print(f"Generated {config['project_dir']}/project.tcl")
    
    # Generate csim.tcl
    with open(f"{config['project_dir']}/csim.tcl", "w") as f:
        f.write(csim_tcl.render(**config))
    print(f"Generated {config['project_dir']}/csim.tcl")
    
    # Generate synth.tcl
    with open(f"{config['project_dir']}/synth.tcl", "w") as f:
        f.write(synth_tcl.render(**config))
    print(f"Generated {config['project_dir']}/synth.tcl")
    
    print(f"\nFixed-Point Configuration:")
    print(f"  Data:   Q{config['data_i']}.{config['data_w']-config['data_i']}")
    print(f"  Weight: Q{config['weight_i']}.{config['weight_w']-config['weight_i']}")
    print(f"  Acc:    Q{config['acc_i']}.{config['acc_w']-config['acc_i']}")
    print(f"  Scale:  Q{config['scale_i']}.{config['scale_w']-config['scale_i']}")
    
    print(f"\nTo run:")
    print(f"  cd {config['project_dir']}")
    print(f"  vitis_hls -f project.tcl  # Create project")
    print(f"  vitis_hls -f csim.tcl     # Run C simulation")
    print(f"  vitis_hls -f synth.tcl    # Run synthesis")

if __name__ == "__main__":
    generate_tcl_scripts()
