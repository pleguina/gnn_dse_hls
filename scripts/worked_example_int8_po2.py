#!/usr/bin/env python3
"""
Worked bit-exact numerical example: INT8 + power-of-two (PO2) GraphSAGE datapath.

Traces ONE real node through BOTH layers of the reduced Cora GraphSAGE model
(Layer 1: Aggregation -> Linear -> ReLU, Layer 2: Aggregation -> Linear ->
identity -> final clamp) through every integer operation performed by the
actual HLS kernel in hls/graphsage_layer_int8_po2.h (functions
aggregate_int8_po2 and linear_int8_po2), and asserts the result is bit-exact
against the reference test-vector generator
(tests/generate_test_vectors_ptq_int8_po2.py) at every stage.

IMPORTANT: this matches the real HLS structure, not a simplified textbook one.
  * Both the aggregation stage and the linear stage FUSE "round + shift +
    saturate-to-INT8" into a single step (exactly like int8_clamp() being
    called right after the shift in the .h file). There is no intermediate
    INT32 value re-exposed after the shift.
  * ReLU is applied AFTER that fused shift+saturate step, i.e. on the INT8
    value, not on the INT32 accumulator. This is mathematically equivalent
    to doing ReLU before saturation because sat8() is symmetric around 0 and
    ReLU is monotonic (ReLU(sat8(x)) == sat8(ReLU(x)) for all x), but the
    HLS code performs it in this order, so the worked example follows suit.
  * This model variant is exported with root_weight=False (see
    src/model_qat.py, src/prepare_ptq_int8_parameters.py): there is only ONE
    branch (the neighbor / aggregation branch). There is no self/root branch
    to combine, so "branch-specific requantization" and "branch combination"
    are N/A for this synthesized variant.

Prerequisites (build/ artifacts, gitignored, regenerate with):
    python3 run_pipeline.py --skip-analysis --steps train,subgraph,quant,int8_ptq,vectors
    cd tests && python3 generate_test_vectors_ptq_int8_po2.py

Usage:
    python3 scripts/worked_example_int8_po2.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INT8_DIR = ROOT / "build/weights_ptq_int8"
PTQ_DIR = ROOT / "build/weights_ptq_float"
TV_DIR = ROOT / "build/test_vectors_ptq_float"
PO2_DIR = ROOT / "build/test_vectors_ptq_int8_po2"

# Node / channels traced through Layer 1 (Aggregation -> Linear -> ReLU) and
# Layer 2 (Aggregation -> Linear -> identity -> final clamp).
NODE = 0        # graph node index in the 8-node HLS test subgraph
IN_CH = 0       # Layer-1 input feature channel used for the aggregation trace
OUT_CH = 0      # Layer-1 hidden channel used for the linear trace
OUT_CH2 = 1     # Layer-2 (final) output/class channel used for the linear trace
                # (chosen because it saturates NEGATIVE, unlike Layer 1's positive
                # saturation, and because Layer 2 uses identity (not ReLU), so a
                # negative accumulator survives instead of being zeroed -- this
                # is the case that actually exercises the "id." branch of rho_l.)


def sat8(x: int) -> int:
    """INT8 saturation, identical to int8_clamp() in graphsage_layer_int8_po2.h."""
    if x > 127:
        return 127
    if x < -128:
        return -128
    return x


def round_shift(x: int, shift: int) -> int:
    """Round-and-shift, identical to the (x + round_const) >> shift used in
    both aggregate_int8_po2() and linear_int8_po2()."""
    round_const = 1 << (shift - 1)
    return (x + round_const) >> shift


def main() -> None:
    for path in (INT8_DIR, PTQ_DIR, TV_DIR, PO2_DIR):
        if not path.exists():
            raise SystemExit(
                f"Missing {path}. Regenerate build/ artifacts first (see module "
                f"docstring for the exact commands)."
            )

    adj = np.loadtxt(INT8_DIR / "adj_matrix_int16.txt", dtype=np.int64)
    x_in = np.loadtxt(TV_DIR / "network_input.txt", dtype=np.int64)
    w1 = np.loadtxt(PTQ_DIR / "conv1_lin_l_weight.txt", dtype=np.int64).reshape(24, 16)
    bias1 = np.loadtxt(INT8_DIR / "bias_layer1_int32.txt", dtype=np.int64)
    w2 = np.loadtxt(PTQ_DIR / "conv2_lin_l_weight.txt", dtype=np.int64).reshape(7, 24)
    bias2 = np.loadtxt(INT8_DIR / "bias_layer2_int32.txt", dtype=np.int64)
    po2cfg = json.loads((PO2_DIR / "po2_config.json").read_text())

    BETA1_SHIFT = po2cfg["shifts"]["BETA1_SHIFT"]
    EFF_SCALE1_SHIFT = po2cfg["shifts"]["EFF_SCALE1_SHIFT"]
    BETA2_SHIFT = po2cfg["shifts"]["BETA2_SHIFT"]
    EFF_SCALE2_SHIFT = po2cfg["shifts"]["EFF_SCALE2_SHIFT"]

    # Reference values from the golden generator, to check bit-exactness against.
    ref_agg1 = np.loadtxt(PO2_DIR / "agg1_int8_po2.txt", dtype=np.int64)
    ref_hidden = np.loadtxt(PO2_DIR / "hidden_int8_po2.txt", dtype=np.int64)
    ref_agg2 = np.loadtxt(PO2_DIR / "agg2_int8_po2.txt", dtype=np.int64)
    ref_output = np.loadtxt(PO2_DIR / "network_output_int8_po2_reference.txt", dtype=np.int64)

    print("=" * 78)
    print(f"WORKED EXAMPLE: INT8-PO2 GraphSAGE, Layer 1, node={NODE}")
    print("=" * 78)

    # ------------------------------------------------------------------
    # Step 1: INT8 input features
    # ------------------------------------------------------------------
    neighbors = [j for j in range(adj.shape[0]) if adj[NODE, j] != 0]
    print(f"\n[1] INT8 input features (node {NODE} and its neighbors, channel f={IN_CH})")
    print(f"    A^K row (adjacency, INT16, K_BITS fixed-point): {adj[NODE].tolist()}")
    print(f"    Neighbors of node {NODE}: {neighbors}")
    for j in neighbors:
        print(f"      x[{j}, {IN_CH}] (INT8) = {x_in[j, IN_CH]}")

    # ------------------------------------------------------------------
    # Step 2 + 3: mean aggregation as integer products/accumulation
    # ------------------------------------------------------------------
    print(f"\n[2/3] Mean aggregation: tmp = sum_j A^K[i,j] * x[j,f]  (integer products + accumulation)")
    terms = []
    tmp = 0
    for j in neighbors:
        a = int(adj[NODE, j])
        xj = int(x_in[j, IN_CH])
        p = a * xj
        terms.append((j, a, xj, p))
        tmp += p
        print(f"      A[{NODE},{j}]={a:6d}  *  x[{j},{IN_CH}]={xj:4d}  =  {p}")
    print(f"    tmp = sum = {tmp}")
    print(f"    (A^K values ~= 4096/{len(neighbors)} = {4096 / len(neighbors):.1f} each -> true mean over "
          f"{len(neighbors)} neighbors in Q{12} fixed point)")

    # ------------------------------------------------------------------
    # Step 4 (part 1) + 6 + 7: PO2 shift with rounding, fused with sat8
    # ------------------------------------------------------------------
    print(f"\n[6/7] PO2 shift (round + shift, shift=BETA1_SHIFT={BETA1_SHIFT}) fused with saturation to INT8")
    round_const_agg = 1 << (BETA1_SHIFT - 1)
    rounded_agg = tmp + round_const_agg
    shifted_agg = rounded_agg >> BETA1_SHIFT
    agg_out = sat8(shifted_agg)
    print(f"    round_const = 2^{BETA1_SHIFT - 1} = {round_const_agg}")
    print(f"    rounded = tmp + round_const = {rounded_agg}")
    print(f"    shifted = rounded >> {BETA1_SHIFT} = {shifted_agg}")
    print(f"    agg_out = sat8(shifted) = {agg_out}   (INT8 aggregation result)")

    assert agg_out == ref_agg1[NODE, IN_CH], (
        f"MISMATCH vs reference agg1_int8_po2.txt: manual={agg_out}, "
        f"reference={ref_agg1[NODE, IN_CH]}"
    )
    print(f"    [OK] bit-exact match vs build/test_vectors_ptq_int8_po2/agg1_int8_po2.txt "
          f"(row {NODE}, col {IN_CH} = {ref_agg1[NODE, IN_CH]})")

    # Full aggregated feature vector for node NODE (needed as input to the linear stage).
    agg_full = ref_agg1[NODE, :].tolist()

    # ------------------------------------------------------------------
    # Step 4 (part 2): bias treatment + integer accumulation (linear layer)
    # ------------------------------------------------------------------
    print(f"\n[4] Bias treatment + linear integer accumulation (output channel o={OUT_CH})")
    print(f"    bias1[{OUT_CH}] (INT32, accumulator domain, i.e. NOT INT8 like the weights) = {bias1[OUT_CH]}")
    print(f"    agg1[node {NODE}, :] (INT8, all 16 channels) = {agg_full}")
    print(f"    weights1[{OUT_CH}, :] (INT8) = {w1[OUT_CH, :].tolist()}")

    acc = int(bias1[OUT_CH])
    print(f"    acc starts at bias1[{OUT_CH}] = {acc}")
    for ff in range(16):
        p = int(agg_full[ff]) * int(w1[OUT_CH, ff])
        acc += p
        print(f"      agg1[{NODE},{ff}]={agg_full[ff]:3d}  *  w1[{OUT_CH},{ff}]={int(w1[OUT_CH, ff]):4d}"
              f"  =  {p:6d}   (acc -> {acc})")
    print(f"    acc (final, INT32 accumulator) = {acc}")

    # ------------------------------------------------------------------
    # Step 5: branch-specific requantization / branch combination
    # ------------------------------------------------------------------
    print(f"\n[5] Branch-specific requantization / branch combination")
    print(f"    N/A for this HLS variant: exported with root_weight=False "
          f"(src/model_qat.py, src/prepare_ptq_int8_parameters.py), so "
          f"graphsage_layer_int8_po2.h has a single neighbor-aggregation branch only "
          f"(linear_int8_po2() takes one 'features' array, no self/root branch to sum).")

    # ------------------------------------------------------------------
    # Step 6/7 (linear stage): PO2 shift with rounding, fused with sat8
    # ------------------------------------------------------------------
    print(f"\n[6/7] PO2 shift (round + shift, shift=EFF_SCALE1_SHIFT={EFF_SCALE1_SHIFT}) "
          f"fused with saturation to INT8")
    round_const_lin = 1 << (EFF_SCALE1_SHIFT - 1)
    rounded_lin = acc + round_const_lin
    shifted_lin = rounded_lin >> EFF_SCALE1_SHIFT
    lin_out = sat8(shifted_lin)
    print(f"    round_const = 2^{EFF_SCALE1_SHIFT - 1} = {round_const_lin}")
    print(f"    rounded = acc + round_const = {rounded_lin}")
    print(f"    shifted = rounded >> {EFF_SCALE1_SHIFT} = {shifted_lin}")
    print(f"    lin_out = sat8(shifted) = {lin_out}   (INT8, BEFORE ReLU)")
    if shifted_lin > 127 or shifted_lin < -128:
        print(f"    -> saturation actually triggered here: {shifted_lin} clamped to {lin_out}")

    # ------------------------------------------------------------------
    # Step 8: output clamp + ReLU (applied on INT8, after saturation; see
    # module docstring for why this order matches the HLS code and is
    # numerically equivalent to ReLU-before-saturate).
    # ------------------------------------------------------------------
    relu_out = max(lin_out, 0)
    print(f"\n[8] rho_l = ReLU (Layer 1 uses ReLU; Layer 2/output uses identity), applied on INT8")
    print(f"    relu_out = max(lin_out, 0) = {relu_out}   (final h_i^(1) value for this channel)")

    assert relu_out == ref_hidden[NODE, OUT_CH], (
        f"MISMATCH vs reference hidden_int8_po2.txt: manual={relu_out}, "
        f"reference={ref_hidden[NODE, OUT_CH]}"
    )
    print(f"    [OK] bit-exact match vs build/test_vectors_ptq_int8_po2/hidden_int8_po2.txt "
          f"(row {NODE}, col {OUT_CH} = {ref_hidden[NODE, OUT_CH]})")

    # ==================================================================
    # LAYER 2: Aggregation -> Linear -> identity -> final output clamp
    # ==================================================================
    print("\n" + "=" * 78)
    print(f"LAYER 2: same node={NODE}, output/class channel o={OUT_CH2}")
    print("(this exercises rho_l = identity, the other branch of Fig. 4's")
    print(" 'ReLU/id.' box, and the network's final output clamp)")
    print("=" * 78)

    hidden_full_node = ref_hidden[NODE, :]  # this node's post-ReLU hidden vector (24 ch)

    # --- Layer-2 aggregation (same neighbor set / adjacency, over hidden features) ---
    print(f"\n[2/3] Layer-2 mean aggregation: tmp2 = sum_j A^K[i,j] * h1[j,{IN_CH}]")
    tmp2 = 0
    for j in neighbors:
        a = int(adj[NODE, j])
        hj = int(ref_hidden[j, IN_CH])
        p = a * hj
        tmp2 += p
        print(f"      A[{NODE},{j}]={a:6d}  *  h1[{j},{IN_CH}]={hj:4d}  =  {p}")
    print(f"    tmp2 = sum = {tmp2}")

    print(f"\n[6/7] PO2 shift (round + shift, shift=BETA2_SHIFT={BETA2_SHIFT}) fused with saturation to INT8")
    round_const_agg2 = 1 << (BETA2_SHIFT - 1)
    rounded_agg2 = tmp2 + round_const_agg2
    shifted_agg2 = rounded_agg2 >> BETA2_SHIFT
    agg2_out = sat8(shifted_agg2)
    print(f"    round_const = 2^{BETA2_SHIFT - 1} = {round_const_agg2}")
    print(f"    rounded = tmp2 + round_const = {rounded_agg2}")
    print(f"    shifted = rounded >> {BETA2_SHIFT} = {shifted_agg2}")
    print(f"    agg2_out = sat8(shifted) = {agg2_out}   (INT8 aggregation result)")

    assert agg2_out == ref_agg2[NODE, IN_CH], (
        f"MISMATCH vs reference agg2_int8_po2.txt: manual={agg2_out}, "
        f"reference={ref_agg2[NODE, IN_CH]}"
    )
    print(f"    [OK] bit-exact match vs build/test_vectors_ptq_int8_po2/agg2_int8_po2.txt "
          f"(row {NODE}, col {IN_CH} = {ref_agg2[NODE, IN_CH]})")

    agg2_full = ref_agg2[NODE, :].tolist()  # full 24-channel aggregated vector for node NODE

    # --- Layer-2 bias treatment + linear accumulation ---
    print(f"\n[4] Bias treatment + linear integer accumulation (output/class channel o={OUT_CH2})")
    print(f"    bias2[{OUT_CH2}] (INT32, accumulator domain) = {bias2[OUT_CH2]}")
    print(f"    agg2[node {NODE}, :] (INT8, all 24 channels) = {agg2_full}")
    print(f"    weights2[{OUT_CH2}, :] (INT8) = {w2[OUT_CH2, :].tolist()}")

    acc2 = int(bias2[OUT_CH2])
    print(f"    acc2 starts at bias2[{OUT_CH2}] = {acc2}")
    for ff in range(24):
        p = int(agg2_full[ff]) * int(w2[OUT_CH2, ff])
        acc2 += p
        print(f"      agg2[{NODE},{ff}]={agg2_full[ff]:4d}  *  w2[{OUT_CH2},{ff}]={int(w2[OUT_CH2, ff]):4d}"
              f"  =  {p:6d}   (acc2 -> {acc2})")
    print(f"    acc2 (final, INT32 accumulator) = {acc2}")

    print(f"\n[5] Branch-specific requantization / branch combination")
    print(f"    N/A (same reason as Layer 1: root_weight=False, single branch).")

    print(f"\n[6/7] PO2 shift (round + shift, shift=EFF_SCALE2_SHIFT={EFF_SCALE2_SHIFT}) "
          f"fused with saturation to INT8")
    round_const_lin2 = 1 << (EFF_SCALE2_SHIFT - 1)
    rounded_lin2 = acc2 + round_const_lin2
    shifted_lin2 = rounded_lin2 >> EFF_SCALE2_SHIFT
    lin2_out = sat8(shifted_lin2)
    print(f"    round_const = 2^{EFF_SCALE2_SHIFT - 1} = {round_const_lin2}")
    print(f"    rounded = acc2 + round_const = {rounded_lin2}")
    print(f"    shifted = rounded >> {EFF_SCALE2_SHIFT} = {shifted_lin2}")
    print(f"    lin2_out = sat8(shifted) = {lin2_out}   (INT8, this IS already the layer's output --")
    print(f"               there is no separate ReLU call for the output layer)")
    if shifted_lin2 > 127 or shifted_lin2 < -128:
        print(f"    -> saturation actually triggered here: {shifted_lin2} clamped to {lin2_out}")

    print(f"\n[8] rho_l = identity (Layer 2 / output layer skips ReLU entirely --")
    print(f"    graphsage_int8_po2_template() calls relu_int8() only after Layer 1's")
    print(f"    linear_int8_po2(), never after Layer 2's) -- so the final network")
    print(f"    output for this class is exactly lin2_out, negative values included:")
    final_out = lin2_out
    print(f"    h_{NODE}^(2)[{OUT_CH2}] = {final_out}")

    assert final_out == ref_output[NODE, OUT_CH2], (
        f"MISMATCH vs reference network_output_int8_po2_reference.txt: "
        f"manual={final_out}, reference={ref_output[NODE, OUT_CH2]}"
    )
    print(f"    [OK] bit-exact match vs "
          f"build/test_vectors_ptq_int8_po2/network_output_int8_po2_reference.txt "
          f"(row {NODE}, col {OUT_CH2} = {ref_output[NODE, OUT_CH2]})")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    rows = [
        ("1. INT8 input features (L1)", f"x[{neighbors},{IN_CH}] = {[int(x_in[j, IN_CH]) for j in neighbors]}"),
        ("2. Mean aggregation (A^K)", f"weights {[int(adj[NODE, j]) for j in neighbors]} (~1/{len(neighbors)} each, Q12)"),
        ("3. Integer products/accum. (L1)", f"tmp = {tmp}"),
        ("4. Bias treatment (L1)", f"bias1[{OUT_CH}] (INT32) = {bias1[OUT_CH]}, acc = {acc}"),
        ("5. Branch-specific requant.", "N/A (root_weight=False, single branch)"),
        ("6. PO2 shift (L1 agg / linear)", f"shift={BETA1_SHIFT} -> {agg_out}; shift={EFF_SCALE1_SHIFT} -> {shifted_lin}"),
        ("7. Saturation (L1, positive)", f"sat8({shifted_lin}) = {lin_out}"),
        ("8. rho_l = ReLU (L1) + clamp", f"N/A branch combo; ReLU(sat8) = {relu_out}"),
        ("-- Layer 2 (output layer) --", ""),
        ("3. Integer products/accum. (L2)", f"tmp2 = {tmp2}"),
        ("4. Bias treatment (L2)", f"bias2[{OUT_CH2}] (INT32) = {bias2[OUT_CH2]}, acc2 = {acc2}"),
        ("6. PO2 shift (L2 agg / linear)", f"shift={BETA2_SHIFT} -> {agg2_out}; shift={EFF_SCALE2_SHIFT} -> {shifted_lin2}"),
        ("7. Saturation (L2, negative)", f"sat8({shifted_lin2}) = {lin2_out}"),
        ("8. rho_l = identity (L2) + clamp", f"N/A branch combo; id(sat8) = {final_out}  <- final network output"),
    ]
    for label, value in rows:
        print(f"  {label:<34s} {value}")

    print("\nAll intermediate and final values verified BIT-EXACT against")
    print("tests/generate_test_vectors_ptq_int8_po2.py output (both layers).")


if __name__ == "__main__":
    main()
