#!/usr/bin/env python3
"""
generate_hls_tcl.py
---------------------

Generate all Vitis HLS TCL scripts (project.tcl, csim.tcl, synth.tcl, cosim.tcl,
ip_export.tcl, clean.tcl) for each module from Jinja2 templates.

This script is called by:
    make -f Makefile.hls hls_setup

It produces one directory per module under build_hls/<module> with:
    project.tcl
    csim.tcl
    synth.tcl
    cosim.tcl
    ip_export.tcl
    clean.tcl

It now supports ALL modules:
    dt_interface
    csc_interface
    concentrator
    layermem
    region_filter
    priority_arbiter
    phi_extrapolation
    phi_dist_processor
    pdf_lookup
    best_candidate
    pdf_lookup_best_candidate
    phidist_pdf_best
    doublebuff
    nn_interface
    regression_nn
"""

import os
import sys
import argparse
import yaml
import re
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

# ---------------------------------------------------------------------------
# Global configuration shared by all modules
# ---------------------------------------------------------------------------

COMMON_PART = "xcvu13p-fsga2577-1-e"    # FPGA part
COMMON_CLOCK_PERIOD = 2.78              # ns (≈360 MHz target)


# ---------------------------------------------------------------------------
# Per-module configuration
#
# For each module we describe:
#   name          : logical module name (also used as folder under build_hls/)
#   top           : top-level C function for Vitis HLS
#   part          : FPGA part (usually COMMON_PART)
#   clock_period  : clock period in ns
#   version       : IP version string
#   vendor        : IP vendor string
#   src           : list of source (.cpp/.hpp) files for synthesis
#   tb            : list of testbench source files OR testbench driver exe source
#   includes      : list of include directories (relative to each module dir)
#   csim_opts     : extra flags passed to csim_design
#   tb_args       : runtime args passed to testbench in csim
#   csynth_opts   : extra flags passed to csynth_design
#   cosim_opts    : extra flags passed to cosim_design
#   cosim_tb_args : runtime args passed to testbench for cosim
#
# EXTRA / OPTIONAL FIELDS:
#   enable_rpc_inputs : (only for concentrator) boolean. If True, we define
#                       -DENABLE_RPC_INPUTS. We automatically inject that into
#                       csim_opts/csynth_opts.
#
# IMPORTANT:
#  * Paths in "src", "tb", and "includes" are LEFT RELATIVE on purpose.
#    They are interpreted by HLS relative to build_hls/<module>.
#
#  * Paths in tb_args / cosim_tb_args that point to data files (*.xml etc.)
#    will be turned into absolute paths because runtime needs absolute.
# ---------------------------------------------------------------------------

MODULE_CONFIGS = {
    "dt_interface": {
        "name": "dt_interface",
        "top": "dt_omtf_interface",
        "part": COMMON_PART,
        "clock_period": COMMON_CLOCK_PERIOD,
        "version": "1.0",
        "vendor": "OMTF",
        "src": [
            "algo/dt_interface/OMTF_dt_interface.cpp"
        ],
        "tb": [
            "algo/dt_interface/OMTF_dt_interface.cpp",
            "verify/tests/tb_dt_interface.cpp",
            "verify/src/xml_parser.cpp",
            "verify/src/xml_model.cpp",
            "verify/src/extractor.cpp",
            "verify/src/logging.cpp",
            "verify/src/latency_shaper.cpp",
            "verify/src/rules.cpp",
            "verify/adapters/csim/mod_dt_interface_adapter.cpp",
            "verify/adapters/cosim/mod_dt_interface_adapter_cosim.cpp"
        ],
        "includes": [
            "algo/common",
            "verify/include"
        ],
        "csim_opts": "-clean -ldflags -lpugixml",
        "tb_args": "--backend csim verify/schemas/data/omtf_events_full.xml",
        "csynth_opts": "",
        "cosim_opts": "-trace_level all -rtl verilog -ldflags -lpugixml",
        "cosim_tb_args": "--backend cosim verify/schemas/data/omtf_events_full.xml",
    },

    "csc_interface": {
        "name": "csc_interface",
        "top": "csc_omtf_interface",
        "part": COMMON_PART,
        "clock_period": COMMON_CLOCK_PERIOD,
        "version": "1.0",
        "vendor": "OMTF",
        "src": [
            "algo/csc_interface/OMTF_csc_interface.cpp"
        ],
        "tb": [
            "algo/csc_interface/OMTF_csc_interface.cpp",
            "verify/tests/tb_csc_interface.cpp",
            "verify/src/xml_parser.cpp",
            "verify/src/xml_model.cpp",
            "verify/src/extractor.cpp",
            "verify/src/logging.cpp",
            "verify/src/latency_shaper.cpp",
            "verify/src/rules.cpp",
            "verify/adapters/csim/mod_csc_interface_adapter.cpp",
            "verify/adapters/cosim/mod_csc_interface_adapter_cosim.cpp"
        ],
        "includes": [
            "algo/common",
            "verify/include"
        ],
        "csim_opts": "-clean -ldflags -lpugixml",
        "tb_args": "--backend csim verify/schemas/data/omtf_events_full.xml",
        "csynth_opts": "",
        "cosim_opts": "-trace_level all -rtl verilog -ldflags -lpugixml",
        "cosim_tb_args": "--backend cosim verify/schemas/data/omtf_events_full.xml",
    },

    "layermem": {
        "name": "layermem",
        "top": "mem",
        "part": COMMON_PART,
        "clock_period": COMMON_CLOCK_PERIOD,
        "version": "1.0",
        "vendor": "OMTF",
        "src": [
            "algo/layermem/OMTF_layerMem.cpp",
            "algo/common/channel_region_layer_lut.h"
        ],
        "tb": [
            "algo/layermem/OMTF_layerMem.cpp",
            "verify/tests/tb_layermem.cpp",
            "verify/src/xml_parser.cpp",
            "verify/src/xml_model.cpp",
            "verify/src/extractor.cpp",
            "verify/src/logging.cpp",
            "verify/src/latency_shaper.cpp",
            "verify/src/rules.cpp",
            "verify/adapters/csim/mod_layermem_adapter.cpp",
            "verify/adapters/cosim/mod_layermem_adapter_cosim.cpp"
        ],
        "includes": [
            "algo/common",
            "verify/include"
        ],
        "csim_opts": "-clean -ldflags -lpugixml",
        "tb_args": "csim verify/schemas/data/omtf_events_full.xml",
        "csynth_opts": "",
        "cosim_opts": "-trace_level all -rtl verilog -ldflags -lpugixml",
        "cosim_tb_args": "cosim verify/schemas/data/omtf_events_full.xml",
    },

    "region_filter": {
        "name": "region_filter",
        "top": "region_filter",
        "part": COMMON_PART,
        "clock_period": COMMON_CLOCK_PERIOD,
        "version": "1.0",
        "vendor": "OMTF",
        "src": [
            "algo/region_filter/OMTF_regionFilter.cpp"
        ],
        "tb": [
            "algo/region_filter/OMTF_regionFilter.cpp",
            "verify/tests/tb_region_filter.cpp",
            "verify/src/xml_parser.cpp",
            "verify/src/xml_model.cpp",
            "verify/src/extractor.cpp",
            "verify/src/logging.cpp",
            "verify/src/latency_shaper.cpp",
            "verify/src/rules.cpp",
            "verify/adapters/csim/mod_region_filter_adapter.cpp",
            "verify/adapters/cosim/mod_region_filter_adapter_cosim.cpp"
        ],
        "includes": [
            "algo/common",
            "verify/include"
        ],
        "csim_opts": "-clean -ldflags -lpugixml",
        "tb_args": "csim verify/schemas/data/omtf_events_full.xml",
        "csynth_opts": "",
        "cosim_opts": "-trace_level all -rtl verilog -ldflags -lpugixml",
        "cosim_tb_args": "--backend cosim verify/schemas/data/omtf_events_full.xml",
    },

    "priority_arbiter": {
        "name": "priority_arbiter",
        "top": "priority_arbiter",
        "part": COMMON_PART,
        "clock_period": COMMON_CLOCK_PERIOD,
        "version": "1.0",
        "vendor": "OMTF",
        "src": [
            "algo/priority_arbiter/OMTF_priority_arbiter.cpp"
        ],
        "tb": [
            "algo/priority_arbiter/OMTF_priority_arbiter.cpp",
            "verify/tests/tb_priority_arbiter.cpp",
            "verify/src/xml_parser.cpp",
            "verify/src/xml_model.cpp",
            "verify/src/extractor.cpp",
            "verify/src/logging.cpp",
            "verify/src/latency_shaper.cpp",
            "verify/src/rules.cpp",
            "verify/adapters/csim/mod_priority_arbiter_adapter.cpp",
            "verify/adapters/cosim/mod_priority_arbiter_adapter_cosim.cpp"
        ],
        "includes": [
            "algo/common",
            "verify/include"
        ],
        "csim_opts": "-clean -ldflags -lpugixml",
        "tb_args": "csim verify/schemas/data/omtf_events_full.xml",
        "csynth_opts": "",
        "cosim_opts": "-trace_level all -rtl verilog -ldflags -lpugixml",
        "cosim_tb_args": "cosim verify/schemas/data/omtf_events_full.xml",
    },

    "phi_extrapolation": {
        "name": "phi_extrapolation",
        "top": "phi_Extrapolator",
        "part": COMMON_PART,
        "clock_period": COMMON_CLOCK_PERIOD,
        "version": "1.0",
        "vendor": "OMTF",
        "src": [
            "algo/phi_extrapolation/OMTF_phi_extrapolator.cpp"
        ],
        "tb": [
            "algo/phi_extrapolation/OMTF_phi_extrapolator.cpp",
            "verify/tests/tb_phi_extrapolation.cpp",
            "verify/src/xml_parser.cpp",
            "verify/src/xml_model.cpp",
            "verify/src/extractor.cpp",
            "verify/src/logging.cpp",
            "verify/src/latency_shaper.cpp",
            "verify/src/rules.cpp",
            "verify/adapters/csim/mod_phi_extrapolation_adapter.cpp",
            "verify/adapters/cosim/mod_phi_extrapolation_adapter_cosim.cpp"
        ],
        "includes": [
            "algo/common",
            "verify/include"
        ],
        "csim_opts": "-clean -ldflags -lpugixml",
        "tb_args": "csim verify/schemas/data/omtf_events_full.xml",
        "csynth_opts": "",
        "cosim_opts": "-trace_level all -rtl verilog -ldflags -lpugixml",
        "cosim_tb_args": "cosim verify/schemas/data/omtf_events_full.xml",
    },

    "phi_dist_processor": {
        "name": "phi_dist_processor",
        "top": "phiDist_processor",
        "part": COMMON_PART,
        "clock_period": COMMON_CLOCK_PERIOD,
        "version": "1.0",
        "vendor": "OMTF",
        "src": [
            "algo/phi_dist_processor/OMTF_phiDist_processor.cpp"
        ],
        "tb": [
            "algo/phi_dist_processor/OMTF_phiDist_processor.cpp",
            "verify/tests/tb_phi_dist_processor.cpp",
            "verify/src/xml_parser.cpp",
            "verify/src/xml_model.cpp",
            "verify/src/extractor.cpp",
            "verify/src/logging.cpp",
            "verify/src/latency_shaper.cpp",
            "verify/src/rules.cpp",
            "verify/adapters/csim/mod_phi_dist_processor_adapter.cpp",
            "verify/adapters/cosim/mod_phi_dist_processor_adapter_cosim.cpp"
        ],
        "includes": [
            "algo/common",
            "verify/include"
        ],
        "csim_opts": "-clean -ldflags -lpugixml",
        "tb_args": "csim verify/schemas/data/DetailedSingleEv_1.xml",
        "csynth_opts": "",
        "cosim_opts": "-trace_level all -rtl verilog -ldflags -lpugixml",
        "cosim_tb_args": "cosim verify/schemas/data/DetailedSingleEv_1.xml",
    },

    "pdf_lookup": {
        "name": "pdf_lookup",
        "top": "pdf_lookup",
        "part": COMMON_PART,
        "clock_period": COMMON_CLOCK_PERIOD,
        "version": "1.0",
        "vendor": "OMTF",
        "src": [
            "algo/pdf_lookup/OMTF_pdf_lookup.cpp"
        ],
        "tb": [
            "algo/pdf_lookup/OMTF_pdf_lookup.cpp",
            "verify/tests/tb_pdf_lookup.cpp",
            "verify/src/xml_parser.cpp",
            "verify/src/xml_model.cpp",
            "verify/src/extractor.cpp",
            "verify/src/logging.cpp",
            "verify/src/latency_shaper.cpp",
            "verify/src/rules.cpp",
            "verify/adapters/csim/mod_pdf_lookup_adapter.cpp",
            "verify/adapters/cosim/mod_pdf_lookup_adapter_cosim.cpp"
        ],
        "includes": [
            "algo/common",
            "verify/include"
        ],
        "csim_opts": "-clean -ldflags -lpugixml",
        "tb_args": "csim verify/schemas/data/DetailedSingleEv_1.xml",
        "csynth_opts": "",
        "cosim_opts": "-trace_level all -rtl verilog -ldflags -lpugixml",
        "cosim_tb_args": "cosim verify/schemas/data/DetailedSingleEv_1.xml",
    },

    "best_candidate": {
        "name": "best_candidate",
        "top": "best_candidate_selector",
        "part": COMMON_PART,
        "clock_period": COMMON_CLOCK_PERIOD,
        "version": "1.0",
        "vendor": "OMTF",
        "src": [
            "algo/best_candidate/OMTF_best_candidate_selector.cpp"
        ],
        "tb": [
            "algo/best_candidate/OMTF_best_candidate_selector.cpp",
            "verify/tests/tb_best_candidate.cpp",
            "verify/src/xml_parser.cpp",
            "verify/src/xml_model.cpp",
            "verify/src/extractor.cpp",
            "verify/src/logging.cpp",
            "verify/src/latency_shaper.cpp",
            "verify/src/rules.cpp",
            "verify/adapters/csim/mod_best_candidate_adapter.cpp",
            "verify/adapters/cosim/mod_best_candidate_adapter_cosim.cpp"
        ],
        "includes": [
            "algo/common",
            "verify/include"
        ],
        "csim_opts": "-clean -ldflags -lpugixml",
        "tb_args": "csim verify/schemas/data/DetailedSingleEv_1.xml",
        "csynth_opts": "",
        "cosim_opts": "-trace_level all -rtl verilog -ldflags -lpugixml",
        "cosim_tb_args": "cosim verify/schemas/data/DetailedSingleEv_1.xml",
    },

    "pdf_lookup_best_candidate": {
        "name": "pdf_lookup_best_candidate",
        "top": "pdf_lookup_best_candidate",
        "part": COMMON_PART,
        "clock_period": COMMON_CLOCK_PERIOD,
        "version": "1.0",
        "vendor": "OMTF",
        "src": [
            "algo/pdf_lookup/OMTF_pdf_lookup.cpp",
            "algo/best_candidate/OMTF_best_candidate_selector.cpp",
            "algo/pdf_lookup/OMTF_pdf_lookup_best_candidate.cpp"
        ],
        "tb": [
            "algo/pdf_lookup/OMTF_pdf_lookup.cpp",
            "algo/best_candidate/OMTF_best_candidate_selector.cpp",
            "algo/pdf_lookup/OMTF_pdf_lookup_best_candidate.cpp",
            "verify/tests/tb_pdf_lookup.cpp",
            "verify/src/xml_parser.cpp",
            "verify/src/xml_model.cpp",
            "verify/src/extractor.cpp",
            "verify/src/logging.cpp",
            "verify/src/latency_shaper.cpp",
            "verify/src/rules.cpp",
            "verify/adapters/csim/mod_pdf_lookup_best_candidate_adapter.cpp",
            "verify/adapters/cosim/mod_pdf_lookup_best_candidate_adapter_cosim.cpp"
        ],
        "includes": [
            "algo/common",
            "verify/include"
        ],
        "csim_opts": "-clean -ldflags -lpugixml",
        "tb_args": "csim verify/schemas/data/DetailedSingleEv_1.xml",
        "csynth_opts": "",
        "cosim_opts": "-trace_level all -rtl verilog -ldflags -lpugixml",
        "cosim_tb_args": "cosim verify/schemas/data/DetailedSingleEv_1.xml",
    },

    "doublebuff": {
        "name": "doublebuff",
        "top": "double_buffer_top",
        "part": COMMON_PART,
        "clock_period": COMMON_CLOCK_PERIOD,
        "version": "1.0",
        "vendor": "OMTF",
        "src": [
            "algo/synth_doublebuff.cpp"
        ],
        "tb": [
            "algo/synth_doublebuff.cpp",
            "algo/test_doublebuff.cpp"
        ],
        "includes": [
            "algo/common"
        ],
        "csim_opts": "-clean",
        "tb_args": "",
        "csynth_opts": "",
        "cosim_opts": "-trace_level all -rtl verilog",
        "cosim_tb_args": "",
    },

    "nn_interface": {
        "name": "nn_interface",
        "top": "NNInterface",
        "part": COMMON_PART,
        "clock_period": COMMON_CLOCK_PERIOD,
        "version": "1.0",
        "vendor": "OMTF",
        "src": [
            "algo/nn_interface/OMTF_nn_interface.cpp"
        ],
        "tb": [],
        "includes": [
            "algo/common"
        ],
        "csim_opts": "",
        "tb_args": "",
        "csynth_opts": "",
        "cosim_opts": "",
        "cosim_tb_args": "",
    },

    "regression_nn": {
        "name": "regression_nn",
        "top": "Regression_NN",
        "part": COMMON_PART,
        "clock_period": COMMON_CLOCK_PERIOD,
        "version": "1.0",
        "vendor": "OMTF",
        "src": [
            "algo/regression_nn/OMTF_regression_nn.cpp"
        ],
        "tb": [],
        "includes": [
            "algo/common"
        ],
        "csim_opts": "",
        "tb_args": "",
        "csynth_opts": "",
        "cosim_opts": "",
        "cosim_tb_args": "",
    },

    "phidist_pdf_best": {
        "name": "phidist_pdf_best",
        "top": "phidist_pdf_best",
        "part": COMMON_PART,
        "clock_period": COMMON_CLOCK_PERIOD,
        "version": "1.0",
        "vendor": "OMTF",
        "src": [
            "algo/phi_dist_processor/OMTF_phiDist_processor.cpp",
            "algo/pdf_lookup/OMTF_pdf_lookup.cpp",
            "algo/best_candidate/OMTF_best_candidate_selector.cpp",
            "algo/OMTF_phidist_pdf_best.cpp"
        ],
        "tb": [],
        "includes": [
            "algo/common"
        ],
        "csim_opts": "",
        "tb_args": "",
        "csynth_opts": "",
        "cosim_opts": "",
        "cosim_tb_args": "",
    },

    "concentrator": {
        "name": "concentrator",
        "top": "subdetector_concentrator",
        "part": COMMON_PART,
        "clock_period": COMMON_CLOCK_PERIOD,
        "version": "1.0",
        "vendor": "OMTF",
        "src": [
            "algo/concentrator/OMTF_concentrator.cpp"
        ],
        "tb": [
            "algo/concentrator/OMTF_concentrator.cpp",
            "verify/tests/tb_concentrator.cpp",
            "verify/src/xml_parser.cpp",
            "verify/src/xml_model.cpp",
            "verify/src/extractor.cpp",
            "verify/src/logging.cpp",
            "verify/src/latency_shaper.cpp",
            "verify/src/rules.cpp",
            "verify/adapters/csim/mod_concentrator_adapter.cpp",
            "verify/adapters/cosim/mod_concentrator_adapter_cosim.cpp"
        ],
        "includes": [
            "algo/common",
            "verify/include"
        ],

        # Special feature flag: if True we will inject
        #    -cflags -DENABLE_RPC_INPUTS
        # into csim_opts and csynth_opts automatically.
        "enable_rpc_inputs": False,

        "csim_opts": "-clean -ldflags -lpugixml",
        "tb_args": "csim verify/schemas/data/omtf_events_full.xml",
        "csynth_opts": "",
        "cosim_opts": "-trace_level all -rtl verilog -ldflags -lpugixml",
        "cosim_tb_args": "cosim verify/schemas/data/omtf_events_full.xml",
    },
}


# ---------------------------------------------------------------------------
# Helper: turn any data file in tb_args / cosim_tb_args into absolute path
# ---------------------------------------------------------------------------

_FILE_ARG_PATTERN = re.compile(r'(\S+\.(?:xml|csv|txt|dat|bin))')

def _abspath_in_args(arg_str: str) -> str:
    """
    Find tokens in arg_str that look like data files (xml/csv/txt/dat/bin),
    and turn them into absolute paths rooted at the project root.

    project_root (repo root) is assumed to be:
        <this_script>/../../..
    i.e.  verify/tools/../.. -> omtf_v2
    """
    if not arg_str:
        return arg_str

    project_root = Path(__file__).parent.parent.parent.absolute()

    def fix_token(match):
        rel_path = match.group(1)

        # if it's already absolute (/home/...), keep it
        if os.path.isabs(rel_path):
            return rel_path

        abs_path = (project_root / rel_path).absolute()

        if not abs_path.exists():
            print(f"⚠️  Warning: File not found: {abs_path}")
        else:
            print(f"📁 Converted to absolute path: {abs_path}")

        return str(abs_path)

    return _FILE_ARG_PATTERN.sub(fix_token, arg_str)


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def _absolutize_data_files(arg_string: str) -> str:
    """
    Take something like:
        "--backend csim verify/schemas/data/foo.xml --event 17"
    and return:
        "--backend csim /abs/path/.../verify/schemas/data/foo.xml --event 17"

    We only absolutize things that LOOK like data files (.xml, .csv, .txt, .dat, .bin).
    We do NOT touch flags like --backend, --event, --latency, etc.
    """
    if not arg_string:
        return arg_string

    file_pattern = re.compile(r'(\S+\.(?:xml|csv|txt|dat|bin))')

    repo_root = Path(__file__).parent.parent.absolute()

    def repl(match):
        rel_path = match.group(1)
        if os.path.isabs(rel_path):
            return rel_path
        abs_path = (repo_root / rel_path).absolute()
        if not abs_path.exists():
            print(f"⚠️  Warning: data file not found: {abs_path}")
        else:
            print(f"📁 Converted to absolute path: {abs_path}")
        return str(abs_path)

    return file_pattern.sub(repl, arg_string)


def _inject_log_dir(arg_string: str, logs_dir_abs: str) -> str:
    """
    Ensure we pass --log <absolute_logs_dir> to the testbench.

    Rules:
    - If arg_string is empty, return empty (some tiny benches might not care).
    - If it already has '--log ...', don't add another.
    - If it starts with '--backend <mode>' we insert [--log <dir>] right after that,
      so ordering is pretty:  --backend csim --log /abs/logs <rest...>
    - Otherwise we just append '--log <dir>' to the end.
    """
    if not arg_string:
        return arg_string

    tokens = arg_string.split()

    # already has --log? leave it alone
    if "--log" in tokens:
        return arg_string

    # pattern: --backend <something> ...
    if len(tokens) >= 2 and tokens[0] == "--backend":
        backend = tokens[0:2]          # ['--backend', 'csim'/'cosim'/...]
        rest    = tokens[2:]           # everything else
        new_tokens = backend + ["--log", logs_dir_abs] + rest
        return " ".join(new_tokens)

    # generic fallback: append
    return arg_string + " --log " + logs_dir_abs


def generate_tcl(module_name, template_dir, output_dir, config_override=None):
    """
    Render all .tcl.j2 templates for one module.
    Writes them into build_hls/<module>/.
    Also:
      - makes all key paths absolute (project_dir, ip_packages_dir, logs_dir)
      - rewrites tb_args / cosim_tb_args to include absolute log dir and absolute data file paths
      - handles enable_rpc_inputs
    """

    # 1. Look up base config
    if module_name not in MODULE_CONFIGS:
        print(f"ERROR: Unknown module '{module_name}'")
        print("Available modules:", ", ".join(MODULE_CONFIGS.keys()))
        return False

    config = MODULE_CONFIGS[module_name].copy()

    # 2. Optional override from YAML
    if config_override:
        config.update(config_override)

    # 3. Handle concentrator RPC mode flag -> inject -DENABLE_RPC_INPUTS into csim/synth flags
    if config.get("enable_rpc_inputs", False):
        rpc_flag = "-cflags -DENABLE_RPC_INPUTS"

        if "csim_opts" in config and config["csim_opts"]:
            config["csim_opts"] = f"{config['csim_opts']} {rpc_flag}"
        else:
            config["csim_opts"] = rpc_flag

        if "csynth_opts" in config and config["csynth_opts"]:
            config["csynth_opts"] = f"{config['csynth_opts']} {rpc_flag}"
        else:
            config["csynth_opts"] = rpc_flag

        print(f"✓ RPC inputs ENABLED for {module_name}")
    elif "enable_rpc_inputs" in config:
        print(f"✓ RPC inputs DISABLED for {module_name} (DT + CSC only)")

    # Don't leak flag into templates
    if "enable_rpc_inputs" in config:
        del config["enable_rpc_inputs"]

    # 4. Absolute important dirs

    # Absolute path to build_hls root
    build_hls_dir = Path(output_dir).absolute()

    # Absolute path to this module's working dir: .../omtf_v2/build_hls/<module>
    module_dir = build_hls_dir / module_name
    module_dir.mkdir(parents=True, exist_ok=True)

    # Per-module absolute logs dir: .../omtf_v2/build_hls/<module>/logs
    logs_dir = module_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Absolute repo root: .../omtf_v2
    repo_root = Path(__file__).parent.parent.absolute()

    # Absolute ip_packages dir under repo root
    ip_packages_dir = repo_root / "ip_packages"
    ip_packages_dir.mkdir(parents=True, exist_ok=True)

    # Convert source files and includes to absolute paths
    # Paths starting with ../ are relative to repo root
    def make_absolute(path_str):
        if Path(path_str).is_absolute():
            return path_str
        # Join and resolve to handle ../ correctly
        abs_path = (repo_root / path_str).resolve()
        return str(abs_path)
    
    if "src" in config:
        config["src"] = [make_absolute(f) for f in config["src"]]
    
    if "tb" in config:
        config["tb"] = [make_absolute(f) for f in config["tb"]]
    
    if "includes" in config:
        config["includes"] = [make_absolute(f) for f in config["includes"]]

    # Add HLS_CSIM_BUILD define for all modules to exclude verilator/xsim adapters
    if "cflags" not in config:
        config["cflags"] = []
    if "-DHLS_CSIM_BUILD" not in config["cflags"]:
        config["cflags"].append("-DHLS_CSIM_BUILD")

    # Store these in config to be available to all templates
    config["project_dir"]     = str(module_dir)
    config["module_name"]     = module_name
    config["project_root"]    = str(repo_root)
    config["ip_packages_dir"] = str(ip_packages_dir)
    config["logs_dir"]        = str(logs_dir)

    # 5. Rewrite tb_args / cosim_tb_args:
    #    - make data files absolute (XML etc.)
    #    - inject --log <abs logs_dir>
    raw_tb_args        = config.get("tb_args", "")
    raw_cosim_tb_args  = config.get("cosim_tb_args", "")

    tb_args_abs        = _absolutize_data_files(raw_tb_args)
    cosim_tb_args_abs  = _absolutize_data_files(raw_cosim_tb_args)

    tb_args_final      = _inject_log_dir(tb_args_abs, str(logs_dir))        if raw_tb_args       else ""
    cosim_tb_args_final= _inject_log_dir(cosim_tb_args_abs, str(logs_dir))  if raw_cosim_tb_args else ""

    config["tb_args"]         = tb_args_final
    config["cosim_tb_args"]   = cosim_tb_args_final

    # 6. Fire up Jinja
    env = Environment(loader=FileSystemLoader(template_dir))

    # 7. Templates we always emit
    templates = [
        "project.tcl.j2",
        "csim.tcl.j2",
        "synth.tcl.j2",
        "cosim.tcl.j2",
        "ip_export.tcl.j2",
        "clean.tcl.j2",
    ]

    # 8. Render each template -> .tcl
    for template_name in templates:
        try:
            template = env.get_template(template_name)
        except Exception as e:
            print(f"❌ Error: could not load template {template_name}: {e}")
            return False

        output_name = template_name.replace(".j2", "")
        output_path = module_dir / output_name

        try:
            rendered = template.render(**config)
        except Exception as e:
            print(f"❌ Error rendering {template_name} for {module_name}: {e}")
            return False

        try:
            with open(output_path, "w") as f:
                f.write(rendered)
        except Exception as e:
            print(f"❌ Error writing {output_path}: {e}")
            return False

        print(f"✅ Generated: {output_path}")

    return True



# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Vitis HLS TCL scripts for OMTF modules."
    )

    parser.add_argument(
        "module",
        nargs="?",
        help=(
            "Module name "
            "(dt_interface, csc_interface, concentrator, layermem, region_filter, "
            "priority_arbiter, phi_extrapolation, phi_dist_processor, pdf_lookup, "
            "best_candidate, doublebuff, or 'all')"
        ),
    )
    parser.add_argument(
        "--template-dir",
        default="../templates",
        help="Directory containing .tcl.j2 templates (default: ../templates)",
    )
    parser.add_argument(
        "--output-dir",
        default="../build_hls",
        help="Directory to place generated module dirs (default: ../build_hls)",
    )
    parser.add_argument(
        "--config",
        help="Optional YAML file with overrides (you can tweak tb_args, enable_rpc_inputs, etc.)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available modules and exit",
    )
    parser.add_argument(
        "--flow",
        choices=["export", "syn", "impl"],
        default="impl",
        help="IP export flow: 'export' (no synth, fastest), 'syn' (RTL synthesis only), or 'impl' (synth + place & route, slowest) (default: impl)",
    )

    args = parser.parse_args()

    # --list shows what modules we currently know about
    if args.list:
        print("Available modules:")
        for name in MODULE_CONFIGS.keys():
            print(f"  - {name}")
        return 0

    # If no module provided, just show usage
    if not args.module:
        parser.print_help()
        return 1

    # Optional YAML override
    config_override = None
    if args.config:
        with open(args.config, "r") as f:
            config_override = yaml.safe_load(f)

    # Add flow parameter to config override
    if config_override is None:
        config_override = {}
    config_override["flow"] = args.flow

    # "all" means iterate every known module in MODULE_CONFIGS
    if args.module == "all":
        modules = list(MODULE_CONFIGS.keys())
    else:
        modules = [args.module]

    ok = True
    for m in modules:
        print(f"\n=== Generating TCL files for {m} ===")
        if not generate_tcl(
            m,
            args.template_dir,
            args.output_dir,
            config_override,
        ):
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
