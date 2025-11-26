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
                       help='Comma-separated steps to run: train,train_qat,subgraph,prune,quant,quant_qat,int8_ptq,vectors,analyze,all')

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

    # Determine Python interpreter path
    python_cmd = os.path.join(script_dir, 'venv', 'bin', 'python')
    if not os.path.exists(python_cmd):
        python_cmd = 'python3'  # Fallback to system python

    # Step 1: Train base and reduced models
    if (run_all or 'train' in steps_to_run) and not args.skip_training:
        success = run_command(
            f"cd src && {python_cmd} train.py",
            "Training Base and Reduced Models"
        )
        if not success:
            return 1

    # Step 1b: Train QAT model
    if (run_all or 'train_qat' in steps_to_run) and not args.skip_training:
        success = run_command(
            f"cd src && {python_cmd} train_qat.py",
            "Training QAT Model (Quantization-Aware Training)"
        )
        if not success:
            print("\n[WARNING] QAT training failed, but continuing...")
            # Don't return, continue with other steps

    # Step 2: Extract subgraph
    if run_all or 'subgraph' in steps_to_run:
        success = run_command(
            f"cd src && {python_cmd} subgraph_extraction.py",
            "Extracting Fixed Subgraph"
        )
        if not success:
            return 1

    # Step 3: Apply pruning (optional)
    if (run_all or 'prune' in steps_to_run) and not args.skip_pruning:
        success = run_command(
            f"cd src && {python_cmd} pruning.py",
            "Applying Structured Pruning"
        )
        if not success:
            print("\n[WARNING] Pruning failed, but continuing...")

    # Step 4: Quantize model with PTQ (Post-Training Quantization)
    if run_all or 'quant' in steps_to_run:
        # Quantize HLS-compatible model (no root_weight)
        success = run_command(
            f"cd src && {python_cmd} quantization_ptq.py",
            "PTQ: Quantizing Model to INT8 (Post-Training Quantization)"
        )
        if not success:
            return 1

        # Quantize standard model (with root_weight)
        success = run_command(
            f"cd src && {python_cmd} quantization_ptq.py --use-root-weight",
            "PTQ: Quantizing Model to INT8 (with root_weight)"
        )
        if not success:
            print("\n[WARNING] PTQ with root_weight failed, but continuing...")

    # Step 4b: Export QAT quantized model
    if run_all or 'quant_qat' in steps_to_run:
        success = run_command(
            f"cd src && {python_cmd} quantization_qat.py",
            "QAT: Exporting Quantized Weights and Test Vectors"
        )
        if not success:
            print("\n[WARNING] QAT export failed, but continuing...")

    # Step 4c: Prepare Integer-Only PTQ Parameters
    if run_all or 'int8_ptq' in steps_to_run:
        # Export float biases from trained model
        success = run_command(
            f"cd src && {python_cmd} export_biases.py",
            "INT8 PTQ: Exporting Float Biases"
        )
        if not success:
            print("\n[WARNING] Float bias export failed, but continuing...")
        
        # Convert to integer-only parameters
        success = run_command(
            f"cd src && {python_cmd} prepare_ptq_int8_parameters.py",
            "INT8 PTQ: Converting to Integer-Only Parameters (M=20, K=4096)"
        )
        if not success:
            print("\n[WARNING] INT8 parameter preparation failed, but continuing...")
        
        # Run integer-only emulator and generate test vectors
        success = run_command(
            f"cd tests && {python_cmd} generate_test_vectors_ptq_int8.py",
            "INT8 PTQ: Running Integer-Only Emulator and Generating Test Vectors"
        )
        if not success:
            print("\n[WARNING] Integer-only emulator failed, but continuing...")

    # Step 5: Generate all test vectors (FLOAT, PTQ, QAT)
    if run_all or 'vectors' in steps_to_run:
        # Generate FLOAT test vectors (for float HLS validation)
        success = run_command(
            f"cd tests && {python_cmd} generate_test_vectors_float.py",
            "Generating FLOAT Test Vectors (Reduced Model - No Quantization)"
        )
        if not success:
            print("\n[WARNING] Float test vector generation failed, but continuing...")

        # Generate PTQ test vectors (quantized)
        success = run_command(
            f"cd tests && {python_cmd} generate_test_vectors_ptq_float.py",
            "Generating PTQ Test Vectors (Post-Training Quantization with Float Ops)"
        )
        if not success:
            return 1

    # Step 6: Analyze models and generate plots
    if (run_all or 'analyze' in steps_to_run) and not args.skip_analysis:
        success = run_command(
            f"cd src && {python_cmd} analyze_models.py",
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
    print("  - build/models/3_reduced_no_root.pth")
    print("  - build/test_vectors_float/ (FLOAT test vectors for HLS)")
    print("\n2. PTQ (POST-TRAINING QUANTIZATION):")
    print("  - build/weights_ptq_float/ (INT8 weights, float quant/dequant)")
    print("  - build/test_vectors_ptq_float/ (PTQ test vectors with float ops)")
    print("\n2b. INTEGER-ONLY PTQ (FIXED-POINT FOR HLS):")
    print("  - build/weights_float/ (original float biases)")
    print("  - build/weights_ptq_int8/ (INT32 biases, INT16 adjacency, fixed-point scales)")
    print("  - build/test_vectors_ptq_int8/ (integer-only test vectors)")
    print("\n3. QAT (QUANTIZATION-AWARE TRAINING):")
    print("  - build/models/4_qat_no_root.pth")
    print("  - build/weights_qat/ (weights and biases)")
    print("  - build/test_vectors_qat/ (test vectors and scales)")
    print("\n4. VISUALIZATION AND ANALYSIS:")
    print("  - build/plots/ (training curves and comparisons)")
    print("  - build/subgraph/")
    print("  - build/hls/weights.h")
    print("\nKey files for FPGA HLS implementation:")
    print("\n  FLOAT HLS:")
    print("    - build/test_vectors_float/")
    print("\n  PTQ FLOAT (dequant→float ops→quant):")
    print("    - build/weights_ptq_float/")
    print("    - build/test_vectors_ptq_float/")
    print("\n  PTQ INT8 (integer-only, no float ops):")
    print("    - build/weights_ptq_int8/")
    print("    - build/test_vectors_ptq_int8/")
    print("\n  QAT (recommended for best accuracy):")
    print("    - build/weights_qat/")
    print("    - build/test_vectors_qat/")
    print("\nNext steps:")
    print("  1. Review training plots in build/plots/")
    print("  2. Compare FLOAT vs PTQ vs INT8-PTQ vs QAT accuracy")
    print("  3a. For FLOAT HLS validation: cd hls && vitis_hls -f run_csim_float.tcl")
    print("  3b. For INT8 HLS implementation (integer-only): Use INT8-PTQ artifacts")
    print("      - No floating-point operations in datapath")
    print("      - Fixed-point arithmetic: M=20, K=4096")
    print("      - Test vectors in build/test_vectors_ptq_int8/")
    print("  3c. For INT8 HLS implementation (QAT): Use QAT artifacts (recommended)")
    print("  4. Run HLS C simulation: cd hls && vitis_hls -f run_csim.tcl")
    print("  5. See INTEGER_PTQ_PIPELINE.md for INT8-PTQ HLS integration")
    print("="*60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
