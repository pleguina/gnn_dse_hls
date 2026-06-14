#!/usr/bin/env python3
"""
Generate TCL files for GraphSAGE QAT v2 HLS implementation.

QAT v2 uses the same pure-integer kernel structure as PTQ-INT8 but with
quantization scales derived from QAT training (fake-quantizer observers).

Scale parameters are loaded at runtime from build/test_vectors_qat_v2/int8_config.txt
so no recompilation is needed when re-exporting weights.

Usage:
  cd hls && python generate_graphsage_qat_v2_tcl.py
  cd ../build/hls/graphsage_qat_v2
  vitis_hls -f project.tcl   # create project + run csim
  vitis_hls -f synth.tcl     # synthesize
"""

from pathlib import Path
from jinja2 import Environment, FileSystemLoader

REPO_ROOT     = Path(__file__).parent.parent.absolute()
HLS_DIR       = REPO_ROOT / "hls"
BUILD_HLS_DIR = REPO_ROOT / "build" / "hls"
PROJECT_DIR   = BUILD_HLS_DIR / "graphsage_qat_v2"
TEMPLATE_DIR  = HLS_DIR / "tcl_example"

config = {
    "module_name": "graphsage_qat_v2",
    "top":         "graphsage_qat_v2",
    "part":        "xcvu13p-fsga2577-1-e",
    "clock_period": 2.77,
    "version":     "1.0",
    "vendor":      "GNN",

    "src": [
        str(HLS_DIR / "graphsage_layer_qat_v2.cpp"),
        str(HLS_DIR / "graphsage_layer_qat_v2.h"),
    ],

    "tb": [
        str(HLS_DIR / "testbench_qat_v2.cpp"),
    ],

    "includes": [
        str(HLS_DIR),
    ],

    # M=24 matches the export default (0 LSB error)
    "cflags": ["-DM_BITS=24"],

    "project_dir":  str(PROJECT_DIR),
    "project_root": str(REPO_ROOT),
    "logs_dir":     str(PROJECT_DIR / "logs"),

    "csim_opts": "-clean",
    "csynth_opts": "",
    "tb_args": "",
}


def main():
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_DIR / "logs").mkdir(parents=True, exist_ok=True)

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))

    for template_name, out_name in [
        ("project.tcl.j2", "project.tcl"),
        ("synth.tcl.j2",   "synth.tcl"),
        ("csim.tcl.j2",    "csim.tcl"),
    ]:
        tmpl = env.get_template(template_name)
        out_path = PROJECT_DIR / out_name
        out_path.write_text(tmpl.render(**config))
        print(f"Generated: {out_path}")

    print(f"\nAll TCL files generated in: {PROJECT_DIR}")
    print(f"\nTo run C-simulation:")
    print(f"  cd {PROJECT_DIR}")
    print(f"  vitis_hls -f project.tcl")
    print(f"\nTo synthesize:")
    print(f"  vitis_hls -f synth.tcl")


if __name__ == "__main__":
    main()
