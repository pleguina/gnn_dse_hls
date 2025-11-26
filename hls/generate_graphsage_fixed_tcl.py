#!/usr/bin/env python3
"""
Generate TCL files for GraphSAGE HLS Fixed-Point implementation.
Uses configurable Q-format fixed-point arithmetic.
"""

import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

# Absolute paths
REPO_ROOT = Path(__file__).parent.parent.absolute()
HLS_DIR = REPO_ROOT / "hls"
BUILD_HLS_DIR = REPO_ROOT / "build" / "hls"
PROJECT_DIR = BUILD_HLS_DIR / "graphsage_fixed"
TEMPLATE_DIR = HLS_DIR / "tcl_example"

# Fixed-point bit widths
DATA_W = 16
DATA_I = 8
WEIGHT_W = 16
WEIGHT_I = 4
ACC_W = 32
ACC_I = 16
SCALE_W = 16
SCALE_I = 2

# Configuration for graphsage_fixed
config = {
    "module_name": "graphsage_fixed",
    "top": "graphsage_network_fixed",
    "part": "xcvu13p-fsga2577-1-e",
    "clock_period": 2.77,
    "version": "1.0",
    "vendor": "GNN",
    
    # Absolute paths to source files
    "src": [
        str(HLS_DIR / "graphsage_layer_fixed.cpp"),
        str(HLS_DIR / "graphsage_layer_fixed.h"),
    ],
    
    # Absolute paths to testbench files
    "tb": [
        str(HLS_DIR / "testbench_fixed.cpp"),
    ],
    
    # Absolute paths to include directories
    "includes": [
        str(HLS_DIR),
    ],
    
    # Additional compiler flags for fixed-point configuration
    "cflags": [
        f"-DDATA_W={DATA_W}",
        f"-DDATA_I={DATA_I}",
        f"-DWEIGHT_W={WEIGHT_W}",
        f"-DWEIGHT_I={WEIGHT_I}",
        f"-DACC_W={ACC_W}",
        f"-DACC_I={ACC_I}",
        f"-DSCALE_W={SCALE_W}",
        f"-DSCALE_I={SCALE_I}",
    ],
    
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
    print(f"\nFixed-Point Configuration:")
    print(f"  Data:   Q{DATA_I}.{DATA_W-DATA_I}")
    print(f"  Weight: Q{WEIGHT_I}.{WEIGHT_W-WEIGHT_I}")
    print(f"  Acc:    Q{ACC_I}.{ACC_W-ACC_I}")
    print(f"  Scale:  Q{SCALE_I}.{SCALE_W-SCALE_I}")
    print(f"\nTo run:")
    print(f"  cd {PROJECT_DIR}")
    print(f"  vitis_hls -f project.tcl  # Create project")
    print(f"  vitis_hls -f csim.tcl     # Run C simulation")
    print(f"  vitis_hls -f synth.tcl    # Run synthesis")

if __name__ == "__main__":
    main()
