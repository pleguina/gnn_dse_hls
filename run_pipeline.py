#!/usr/bin/env python3
"""
Main pipeline script to run the entire GNN-to-FPGA workflow.
Executes all steps from training to test vector generation.
"""

import os
import sys
import subprocess
import argparse


def run_command(cmd, description):
    """Run a command and print status."""
    print("\n" + "="*60)
    print(f"Step: {description}")
    print("="*60)
    print(f"Running: {cmd}\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed!")
        return False
    else:
        print(f"\n[SUCCESS] {description} completed!")
        return True


def main():
    parser = argparse.ArgumentParser(description='Run GraphSAGE FPGA pipeline')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training (use existing models)')
    parser.add_argument('--skip-pruning', action='store_true',
                       help='Skip pruning step')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip model analysis and plotting')
    parser.add_argument('--steps', type=str, default='all',
                       help='Comma-separated steps to run: train,subgraph,prune,quant,vectors,analyze,all')

    args = parser.parse_args()

    # Parse steps
    steps_to_run = set(args.steps.split(','))
    run_all = 'all' in steps_to_run

    # Change to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("\n" + "="*60)
    print("GraphSAGE FPGA Implementation Pipeline")
    print("="*60)
    print(f"Working directory: {os.getcwd()}")

    success = True

    # Step 1: Train models
    if (run_all or 'train' in steps_to_run) and not args.skip_training:
        success = run_command(
            "cd src && python3 train.py",
            "Training Base and Reduced Models"
        )
        if not success:
            return 1

    # Step 2: Extract subgraph
    if run_all or 'subgraph' in steps_to_run:
        success = run_command(
            "cd src && python3 subgraph_extraction.py",
            "Extracting Fixed Subgraph"
        )
        if not success:
            return 1

    # Step 3: Apply pruning (optional)
    if (run_all or 'prune' in steps_to_run) and not args.skip_pruning:
        success = run_command(
            "cd src && python3 pruning.py",
            "Applying Structured Pruning"
        )
        if not success:
            print("\n[WARNING] Pruning failed, but continuing...")

    # Step 4: Quantize model (both versions)
    if run_all or 'quant' in steps_to_run:
        # Quantize HLS-compatible model (no root_weight)
        success = run_command(
            "cd src && python3 quantization.py",
            "Quantizing Model to INT8 (HLS-compatible, no root_weight)"
        )
        if not success:
            return 1
        
        # Quantize standard model (with root_weight)
        success = run_command(
            "cd src && python3 quantization.py --use-root-weight",
            "Quantizing Model to INT8 (with root_weight)"
        )
        if not success:
            print("\n[WARNING] Quantization with root_weight failed, but continuing...")

    # Step 5: Generate test vectors
    if run_all or 'vectors' in steps_to_run:
        success = run_command(
            "cd tests && python3 generate_test_vectors.py",
            "Generating Test Vectors for HLS"
        )
        if not success:
            return 1

    # Step 6: Analyze models and generate plots
    if (run_all or 'analyze' in steps_to_run) and not args.skip_analysis:
        success = run_command(
            "cd src && python3 analyze_models.py",
            "Analyzing Models and Generating Plots"
        )
        if not success:
            print("\n[WARNING] Analysis failed, but continuing...")

    # Final summary
    print("\n" + "="*60)
    print("Pipeline Execution Complete!")
    print("="*60)
    print("\nGenerated artifacts:")
    print("  - build/models/base_graphsage_best.pth")
    print("  - build/models/reduced_graphsage_best.pth")
    print("  - build/subgraph/")
    print("  - build/quantized/")
    print("  - build/plots/ (visualization plots)")
    print("  - build/hls/weights.h")
    print("  - build/test_vectors/")
    print("\nVisualization plots:")
    print("  - build/plots/base_model_training.png")
    print("  - build/plots/reduced_model_training.png")
    print("  - build/plots/model_comparison.png")
    print("  - build/plots/efficiency_analysis.png")
    print("  - build/plots/accuracy_degradation.png")
    print("  - build/plots/summary_report.png")
    print("\nNext steps:")
    print("  1. Review plots in build/plots/")
    print("  2. Check build/plots/model_stats.json for detailed metrics")
    print("  3. Run HLS C simulation: cd hls && vitis_hls -f run_csim.tcl")
    print("  4. See README.md for detailed usage")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
