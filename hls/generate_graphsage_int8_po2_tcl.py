#!/usr/bin/env python3
"""
Generate TCL files for GraphSAGE HLS INT8 PO2 (Power-of-Two Scales) implementation.
This is the hardware-optimized version with shift-based scaling instead of multipliers.

Key features:
- No DSP48 usage for scale multiplications (just bit-shifts)
- Compile-time shift constants (no runtime scale parameters)
- Shorter critical path than arbitrary-scale INT8
- Same M=24 fractional bits for comparison with INT8

Trade-off:
- Small accuracy loss due to power-of-two scale approximation
- Significant resource savings (hundreds of DSPs saved)
"""

import json
import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

# Absolute paths
REPO_ROOT = Path(__file__).parent.parent.absolute()
HLS_DIR = REPO_ROOT / "hls"
BUILD_HLS_DIR = REPO_ROOT / "build" / "hls"
PROJECT_DIR = BUILD_HLS_DIR / "graphsage_int8_po2"
TEMPLATE_DIR = HLS_DIR / "tcl_example"

# Load calibrated PO2 shift values from test vector config
PO2_CONFIG_PATH = REPO_ROOT / "build" / "test_vectors_ptq_int8_po2" / "po2_config.json"
shifts = {"BETA1_SHIFT": 17, "BETA2_SHIFT": 12, "EFF_SCALE1_SHIFT": 6, "EFF_SCALE2_SHIFT": 6}
if PO2_CONFIG_PATH.exists():
    with open(PO2_CONFIG_PATH) as f:
        po2_cfg = json.load(f)
    shifts.update(po2_cfg.get("shifts", {}))

cflags = [
    "-DM_BITS=24",
    f"-DBETA1_SHIFT={shifts['BETA1_SHIFT']}",
    f"-DBETA2_SHIFT={shifts['BETA2_SHIFT']}",
    f"-DEFF_SCALE1_SHIFT={shifts['EFF_SCALE1_SHIFT']}",
    f"-DEFF_SCALE2_SHIFT={shifts['EFF_SCALE2_SHIFT']}",
]

# Configuration for graphsage_int8_po2
config = {
    "module_name": "graphsage_int8_po2",
    "top": "graphsage_int8_po2",
    "part": "xcvu13p-fsga2577-1-e",
    "clock_period": 2.77,
    "version": "1.0",
    "vendor": "GNN",

    # Absolute paths to source files
    "src": [
        str(HLS_DIR / "graphsage_layer_int8_po2.cpp"),
        str(HLS_DIR / "graphsage_layer_int8_po2.h"),
    ],

    # Absolute paths to testbench files
    "tb": [
        str(HLS_DIR / "testbench_int8_po2.cpp"),
    ],

    # Absolute paths to include directories
    "includes": [
        str(HLS_DIR),
    ],

    # Shift values read from po2_config.json (calibrated per model)
    "cflags": cflags,
    
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
    print(f"\n--- INT8 PO2 (Power-of-Two Scales) Version ---")
    print(f"Key benefit: No DSP48s for scale multiplication (just bit-shifts)")
    print(f"Trade-off: Small accuracy loss due to PO2 approximation")
    print(f"\nTo run:")
    print(f"  cd {PROJECT_DIR}")
    print(f"  vitis_hls -f project.tcl  # Create project")
    print(f"  vitis_hls -f csim.tcl     # Run C simulation")
    print(f"  vitis_hls -f synth.tcl    # Run synthesis")

if __name__ == "__main__":
    main()
