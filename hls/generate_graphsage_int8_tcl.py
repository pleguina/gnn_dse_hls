#!/usr/bin/env python3
"""
Generate TCL files for GraphSAGE HLS INT8 (Pure Integer) implementation.
This is the hardware-optimized version with integer-only arithmetic.

Key features:
- M=24 fractional bits for exact match with PTQ-Float (0 LSB error)
- Pure INT8 weights and activations
- INT32 accumulators
- INT16 adjacency matrix (scaled by K=4096)
"""

import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

# Absolute paths
REPO_ROOT = Path(__file__).parent.parent.absolute()
HLS_DIR = REPO_ROOT / "hls"
BUILD_HLS_DIR = REPO_ROOT / "build" / "hls"
PROJECT_DIR = BUILD_HLS_DIR / "graphsage_int8"
TEMPLATE_DIR = HLS_DIR / "tcl_example"

# Configuration for graphsage_int8
config = {
    "module_name": "graphsage_int8",
    "top": "graphsage_int8",
    "part": "xcvu13p-fsga2577-1-e",
    "clock_period": 2.77,
    "version": "1.0",
    "vendor": "GNN",
    
    # Absolute paths to source files
    "src": [
        str(HLS_DIR / "graphsage_layer_int8.cpp"),
        str(HLS_DIR / "graphsage_layer_int8.h"),
    ],
    
    # Absolute paths to testbench files
    "tb": [
        str(HLS_DIR / "testbench_int8.cpp"),
    ],
    
    # Absolute paths to include directories
    "includes": [
        str(HLS_DIR),
    ],
    
    # Additional compiler flags (M_BITS=24 for exact match with PTQ-Float)
    "cflags": ["-DM_BITS=24"],
    
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
