# Worked bit-exact numerical example: INT8 + power-of-two (PO2) datapath

This is a worked numerical integer-quantization example that traces a single
node/channel through: (1) INT8 input features, (2) mean aggregation,
(3) integer products/accumulation, (4) bias treatment, (5) branch-specific
requantization, (6) PO2 shift, (7) saturation, (8) branch combination and
output clamp.

This document traces a real node and channel from the trained Cora model,
using the actual arithmetic performed by the synthesized HLS kernel
(`hls/graphsage_layer_int8_po2.h`, functions `aggregate_int8_po2` and
`linear_int8_po2`), not a simplified or hand-picked toy example. It
corresponds to the `Aggregation -> S_beta shift+sat8 -> Linear -> S_gamma
shift+sat8 -> rho_l ReLU/id.` datapath of Fig. 4 and Eqs. 5-8 in the main
paper.

## Reproducing this example

```bash
# 1. Generate the trained model + INT8-PTQ artifacts
python3 run_pipeline.py --skip-analysis --steps train,subgraph,quant,int8_ptq,vectors

# 2. Generate the PO2 (power-of-two shift) test vectors
cd tests && python3 generate_test_vectors_ptq_int8_po2.py && cd ..

# 3. Run the worked example (prints every step below and asserts bit-exactness)
python3 scripts/worked_example_int8_po2.py
```

The script re-derives every intermediate integer value by hand from the same
`build/` artifacts the HLS testbench consumes, and asserts that its result
equals the reference generator's output
(`build/test_vectors_ptq_int8_po2/agg1_int8_po2.txt` and
`hidden_int8_po2.txt`). So the example is verified bit-exact, not just
internally consistent.

## Structural note: this follows the real HLS code, not a simplified textbook order

Both the aggregation stage and the linear stage in
`graphsage_layer_int8_po2.h` fuse "round + shift + saturate-to-INT8" into a
single step (`int8_clamp()` is called immediately after the shift). There is
no separately-exposed INT32 value after the shift. ReLU is applied
afterwards, on the already-saturated INT8 value (`relu_int8()` operates on
the INT8 array), not on the INT32 accumulator.

This is numerically equivalent to applying ReLU before saturation, because
`sat8()` is symmetric around 0 and ReLU is monotonic
(`ReLU(sat8(x)) == sat8(ReLU(x))` for all `x`), so the results don't differ.
Still, the worked example below follows the code's actual order (shift+sat8,
then ReLU) rather than the order shown in an earlier draft of Fig. 4.

## Single-branch caveat (root_weight = False)

Items 5 and 8 above assume a self/root branch combined with the neighbor
branch, as in generic PyG `SAGEConv`
(`out = W_l * mean(x_neigh) + W_r * x_self + b`). The HLS variant actually
synthesized in this repository is exported with `root_weight=False` (see
`src/model_qat.py`, `src/prepare_ptq_int8_parameters.py`):

```
out = W_l * mean_aggregate(x_neigh) + b
```

`linear_int8_po2()` therefore takes a single `features` array, so there is
no second branch to requantize or combine. The worked example marks steps 5
and 8 as N/A for this reason, rather than inventing a self-branch that does
not exist in the synthesized design.

## Scope: both layers, both branches of rho_l

The example traces node 0 through both layers so it exercises both values of
rho_l shown in Fig. 4 (`ReLU/id.`): Layer 1 uses ReLU, Layer 2 (the output
layer) uses identity. This also closes out step 8 ("output clamp") against
the network's actual final output, not just an intermediate hidden state:
`graphsage_int8_po2_template()` calls `relu_int8()` only after
Layer 1's `linear_int8_po2()`, never after Layer 2's, so Layer 2's output is
simply `sat8(shift(acc))`.

The Layer-2 output channel (`o=1`) was deliberately chosen because it
saturates on the negative side (`-128`), unlike Layer 1's channel, which
saturated positive (`127`). This is the case that actually distinguishes the
identity branch from ReLU: had this been routed through ReLU instead, a
negative accumulator would have been zeroed. Here it correctly survives all
the way to the final clamp.

## Worked example: node 0, Layer 1 (Aggregation -> Linear -> ReLU)

Node 0 has 3 neighbors in the 8-node HLS test subgraph (nodes 1, 5, 7),
which makes the mean-aggregation step non-trivial, and its Layer-1 linear
output happens to overflow the INT8 range, giving a genuine (not
artificially forced) example of saturation.

### 1. INT8 input features

Feature channel `f=0` of node 0 and its neighbors:

| node j | x[j, 0] (INT8) |
|---|---|
| 1 | 38 |
| 5 | 32 |
| 7 | 46 |

### 2. Mean aggregation (A^K, INT16, Q12 fixed point)

Adjacency row for node 0: `[0, 1365, 0, 0, 0, 1365, 0, 1365]`.
Each nonzero weight is `1365 ≈ 4096/3` (`K_BITS=12`, i.e. `4096 = 1.0` in
fixed point), which is the mean over the 3 neighbors, expressed as an
integer.

### 3. Integer products / accumulation

```
tmp = A[0,1]*x[1,0] + A[0,5]*x[5,0] + A[0,7]*x[7,0]
    = 1365*38      + 1365*32      + 1365*46
    = 51870        + 43680        + 62790
    = 158340
```

### 6/7. PO2 shift (aggregation) fused with saturation

```
shift        = BETA1_SHIFT = 16
round_const  = 2^15 = 32768
rounded      = 158340 + 32768 = 191108
shifted      = 191108 >> 16   = 2
agg_out      = sat8(2)        = 2      (INT8, no saturation needed here)
```

`agg_out = 2` matches `agg1_int8_po2.txt[0, 0]` exactly.

The full aggregated feature vector for node 0 (all 16 channels, same
procedure applied per channel) is:

```
agg1[0, :] = [2, 3, 2, 3, 3, 4, 2, 5, 2, 2, 2, 4, 2, 3, 3, 2]
```

### 4. Bias treatment + linear integer accumulation (output channel o=0)

The bias lives in the accumulator domain (`bias_t = ap_int<ACC_BITS>`, the
"INT32" path in Fig. 4), not in INT8 like the weights, so it is added
directly to the INT32 accumulator without requantization:

```
bias1[0] (INT32) = 107777
acc = bias1[0] + sum_f( agg1[0,f] * weights1[0,f] )
    = 107777 + (-30 -201 -130 +51 -168 +160 -128 +250 +58 +170 +94 -252 +132 -9 +150 +154)
    = 107777 + 301
    = 108078
```

### 5. Branch-specific requantization / branch combination

N/A: `root_weight=False`, single neighbor branch only (see caveat above).

### 6/7. PO2 shift (linear) fused with saturation

```
shift        = EFF_SCALE1_SHIFT = 7
round_const  = 2^6 = 64
rounded      = 108078 + 64 = 108142
shifted      = 108142 >> 7 = 844
lin_out      = sat8(844)   = 127      <-- saturation triggered (844 > 127)
```

### 8. rho_l (ReLU / identity) + output clamp

Layer 1 uses ReLU (Layer 2 / final output uses identity, see
`graphsage_int8_po2_template()`), applied on the already-saturated INT8
value:

```
relu_out = max(lin_out, 0) = max(127, 0) = 127
```

`relu_out = 127` matches `hidden_int8_po2.txt[0, 0]` exactly. This is
`h_0^{(1)}` for channel 0.

## Worked example: node 0, Layer 2 (Aggregation -> Linear -> identity -> final clamp)

Layer 2 aggregates the same node's post-ReLU hidden vector (`h1`, 24
channels) over the same neighbor set (nodes 1, 5, 7), using the
independently-fit `BETA2_SHIFT`, then applies `linear_int8_po2()` with
`weights2`/`bias2` and `EFF_SCALE2_SHIFT` to produce class channel `o=1`.

### 2/3. Mean aggregation + integer accumulation (channel f=0)

```
tmp2 = A[0,1]*h1[1,0] + A[0,5]*h1[5,0] + A[0,7]*h1[7,0]
     = 1365*127       + 1365*127       + 1365*127
     = 520065
```
All three neighbors' Layer-1 hidden channel 0 happen to be saturated at
`127` for this run (see script output for the exact per-term breakdown), so
`agg2[0,0] = 127`, matching `agg2_int8_po2.txt[0,0]` bit-exact.

### 4. Bias treatment + linear integer accumulation (channel o=1)

```
bias2[1] (INT32) = -95541
acc2 = bias2[1] + sum_f( agg2[0,f] * weights2[1,f] )   (24 terms, see script output)
     = -122217
```

### 5. Branch-specific requantization / branch combination

N/A: same reason as Layer 1 (`root_weight=False`).

### 6/7. PO2 shift (linear) fused with saturation

```
shift        = EFF_SCALE2_SHIFT = 7
round_const  = 2^6 = 64
rounded      = -122217 + 64 = -122153
shifted      = -122153 >> 7 = -955
lin2_out     = sat8(-955)   = -128     <-- saturation triggered, negative side
```

### 8. rho_l = identity + final output clamp

Layer 2 has no ReLU call in `graphsage_int8_po2_template()`, so its output
is `lin2_out` directly:

```
h_0^{(2)}[1] = -128
```

`h_0^{(2)}[1] = -128` matches `network_output_int8_po2_reference.txt[0, 1]`
exactly. This is the network's actual final output for class 1 at node 0.
Because identity (not ReLU) is applied here, the negative accumulator
correctly survives to the output instead of being zeroed.

## Summary table

| # | Step | Layer 1 (rho_l = ReLU) | Layer 2 / output (rho_l = identity) |
|---|---|---|---|
| 1 | INT8 input features | `x[{1,5,7}, 0] = [38, 32, 46]` | `h1[{1,5,7}, 0]` (Layer-1 output) |
| 2 | Mean aggregation (A^K) | weights `[1365, 1365, 1365]` ≈ 1/3 each, Q12 | same adjacency row |
| 3 | Integer products/accumulation | `tmp = 158340` | `tmp2 = 520065` |
| 4 | Bias treatment | `bias1[0] (INT32) = 107777` -> `acc = 108078` | `bias2[1] (INT32) = -95541` -> `acc2 = -122217` |
| 5 | Branch-specific requantization | N/A (`root_weight=False`, single branch) | N/A (same reason) |
| 6 | PO2 shift | agg: shift=16 -> `2`; linear: shift=7 -> `844` | agg: shift=12 -> `127`; linear: shift=7 -> `-955` |
| 7 | Saturation | `sat8(844) = 127` (positive) | `sat8(-955) = -128` (negative) |
| 8 | Branch combination + output clamp | N/A branch combo; `ReLU(127) = 127` | N/A branch combo; `id(-128) = -128` (final network output) |

Verification: every intermediate value above (`agg1[0,0]=2`,
`hidden[0,0]=127`, `agg2[0,0]=127`, `output[0,1]=-128`) is checked by
`scripts/worked_example_int8_po2.py` with four separate `assert` statements,
bit-exact against the independent reference implementation in
`tests/generate_test_vectors_ptq_int8_po2.py`.
