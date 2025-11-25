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
                       help='Comma-separated steps to run: train,train_qat,subgraph,prune,quant,quant_qat,vectors,analyze,all')

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

    # Step 1: Train base and reduced models
    if (run_all or 'train' in steps_to_run) and not args.skip_training:
        success = run_command(
            "cd src && python3 train.py",
            "Training Base and Reduced Models"
        )
        if not success:
            return 1

    # Step 1b: Train QAT model
    if (run_all or 'train_qat' in steps_to_run) and not args.skip_training:
        success = run_command(
            "cd src && python3 train_qat.py",
            "Training QAT Model (Quantization-Aware Training)"
        )
        if not success:
            print("\n[WARNING] QAT training failed, but continuing...")
            # Don't return, continue with other steps

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

    # Step 4: Quantize model with PTQ (Post-Training Quantization)
    if run_all or 'quant' in steps_to_run:
        # Quantize HLS-compatible model (no root_weight)
        success = run_command(
            "cd src && python3 quantization.py",
            "PTQ: Quantizing Model to INT8 (Post-Training Quantization)"
        )
        if not success:
            return 1

        # Quantize standard model (with root_weight)
        success = run_command(
            "cd src && python3 quantization.py --use-root-weight",
            "PTQ: Quantizing Model to INT8 (with root_weight)"
        )
        if not success:
            print("\n[WARNING] PTQ with root_weight failed, but continuing...")

    # Step 4b: Export QAT quantized model
    if run_all or 'quant_qat' in steps_to_run:
        success = run_command(
            "cd src && python3 quantization_qat.py",
            "QAT: Exporting Quantized Weights and Test Vectors"
        )
        if not success:
            print("\n[WARNING] QAT export failed, but continuing...")

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
    print("\n1. FLOAT BASELINE MODEL:")
    print("  - build/models/base_graphsage_best.pth")
    print("  - build/models/reduced_graphsage_no_root_best.pth")
    print("\n2. PTQ (POST-TRAINING QUANTIZATION):")
    print("  - build/quantized/ (weights and test vectors)")
    print("  - build/test_vectors/")
    print("\n3. QAT (QUANTIZATION-AWARE TRAINING):")
    print("  - build/models/reduced_graphsage_qat_no_root_best.pth")
    print("  - build/quantized_qat/ (weights and biases)")
    print("  - build/test_vectors_qat/ (test vectors and scales)")
    print("\n4. VISUALIZATION AND ANALYSIS:")
    print("  - build/plots/ (training curves and comparisons)")
    print("  - build/subgraph/")
    print("  - build/hls/weights.h")
    print("\nKey files for FPGA HLS implementation (QAT recommended):")
    print("  - build/quantized_qat/weights_layer{1,2}_qat.txt")
    print("  - build/quantized_qat/bias_layer{1,2}_qat.txt")
    print("  - build/test_vectors_qat/network_input_qat.txt")
    print("  - build/test_vectors_qat/network_output_reference_qat.txt")
    print("  - build/test_vectors_qat/edge_index_qat.txt")
    print("  - build/test_vectors_qat/scales_qat.txt")
    print("\nNext steps:")
    print("  1. Review training plots in build/plots/")
    print("  2. Compare FLOAT vs PTQ vs QAT accuracy")
    print("  3. Use QAT artifacts for HLS implementation (recommended)")
    print("  4. Run HLS C simulation: cd hls && vitis_hls -f run_csim.tcl")
    print("  5. See QAT_scope.txt for HLS integration details")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
