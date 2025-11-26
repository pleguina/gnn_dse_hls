#!/usr/bin/env python3
"""
Generate TCL files for GraphSAGE HLS INT8 PTQ (Post-Training Quantization) implementation
"""

import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

# Absolute paths
REPO_ROOT = Path(__file__).parent.parent.absolute()
HLS_DIR = REPO_ROOT / "hls"
BUILD_HLS_DIR = REPO_ROOT / "build" / "hls"
PROJECT_DIR = BUILD_HLS_DIR / "graphsage_ptq"
TEMPLATE_DIR = HLS_DIR / "tcl_example"

# Configuration for graphsage_ptq (INT8 PTQ)
config = {
    "module_name": "graphsage_ptq",
    "top": "graphsage_network_ptq",
    "part": "xcvu13p-fsga2577-1-e",
    "clock_period": 2.77,
    "version": "1.0",
    "vendor": "GNN",
    
    # Absolute paths to source files
    "src": [
        str(HLS_DIR / "graphsage_layer_ptq.cpp"),
        str(HLS_DIR / "graphsage_layer_ptq.h"),
    ],
    
    # Absolute paths to testbench files
    "tb": [
        str(HLS_DIR / "testbench_ptq.cpp"),
    ],
    
    # Absolute paths to include directories
    "includes": [
        str(HLS_DIR),
    ],
    
    # Additional compiler flags
    "cflags": [],
    
    # Absolute paths
    "project_dir": str(PROJECT_DIR),
    "project_root": str(REPO_ROOT),
    "logs_dir": str(PROJECT_DIR / "logs"),
    
    # TCL options
    "csim_opts": "-clean",
    "csynth_opts": "",
    "tb_args": "",
}

def main():
    # Create project directory
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    
    # Setup Jinja environment
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    
    # Generate project.tcl
    template = env.get_template("project.tcl.j2")
    output_path = PROJECT_DIR / "project.tcl"
    with open(output_path, "w") as f:
        f.write(template.render(**config))
    print(f"✅ Generated: {output_path}")
    
    # Generate synth.tcl
    template = env.get_template("synth.tcl.j2")
    output_path = PROJECT_DIR / "synth.tcl"
    with open(output_path, "w") as f:
        f.write(template.render(**config))
    print(f"✅ Generated: {output_path}")
    
    # Generate csim.tcl
    template = env.get_template("csim.tcl.j2")
    output_path = PROJECT_DIR / "csim.tcl"
    with open(output_path, "w") as f:
        f.write(template.render(**config))
    print(f"✅ Generated: {output_path}")
    
    print(f"\n✅ All TCL files generated in: {PROJECT_DIR}")
    print(f"\nTo run:")
    print(f"  cd {PROJECT_DIR}")
    print(f"  vitis_hls -f project.tcl  # Create project")
    print(f"  vitis_hls -f csim.tcl     # Run C simulation")
    print(f"  vitis_hls -f synth.tcl    # Run synthesis")

if __name__ == "__main__":
    main()
