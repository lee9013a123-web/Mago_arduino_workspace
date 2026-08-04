# CAM++ static graph report

- model: `models/campplus_int8_static_qop.onnx`
- opset: 11, input `feature` -> output `embedding`
- raw graph: **3180 nodes**, 1963 initializers

## Length buckets

| bucket | fbank frames | nodes | runtime nodes | folded away | MACs | arena | weights |
|---|---|---|---|---|---|---|---|
| 1 s | 98 | 3180 | 1438 | 1742 (55%) | 552 M | 1.91 MB | 7.08 MB |
| 3 s | 298 | 3180 | 1438 | 1742 (55%) | 1677 M | 5.82 MB | 7.08 MB |
| 5 s | 498 | 3180 | 1438 | 1742 (55%) | 2803 M | 9.73 MB | 7.08 MB |
| 10 s | 998 | 3180 | 1438 | 1742 (55%) | 5617 M | 19.49 MB | 7.08 MB |

`folded away` = nodes whose inputs are all compile-time known once the input length is frozen (Shape/Gather/Concat shape arithmetic, Constants). A static engine never executes them.

## Op histogram (raw vs static)

At 3 s / 298 frames.

| op | raw | after static shape resolution | removed |
|---|---|---|---|
| Constant | 580 | 0 | 580 |
| Unsqueeze | 369 | 54 | 315 |
| Shape | 265 | 0 | 265 |
| QLinearConv | 225 | 225 | 0 |
| DequantizeLinear | 225 | 225 | 0 |
| QuantizeLinear | 223 | 223 | 0 |
| Gather | 213 | 0 | 213 |
| Relu | 171 | 171 | 0 |
| Concat | 158 | 53 | 105 |
| Mul | 107 | 54 | 53 |
| Reshape | 105 | 53 | 52 |
| Add | 56 | 56 | 0 |
| BatchNormalization | 56 | 56 | 0 |
| ReduceMean | 55 | 55 | 0 |
| AveragePool | 52 | 52 | 0 |
| ConstantOfShape | 52 | 0 | 52 |
| Equal | 52 | 0 | 52 |
| Where | 52 | 0 | 52 |
| Expand | 52 | 52 | 0 |
| Slice | 52 | 52 | 0 |
| Sigmoid | 52 | 52 | 0 |
| Sub | 2 | 1 | 1 |
| Transpose | 1 | 1 | 0 |
| ReduceProd | 1 | 0 | 1 |
| Cast | 1 | 0 | 1 |
| Div | 1 | 1 | 0 |
| Sqrt | 1 | 1 | 0 |
| Squeeze | 1 | 1 | 0 |
| **total** | **3180** | **1438** | **1742** |

## Where the compute is

At 3 s, total 1677 M MAC.

| scope | nodes | runtime | MACs | share | weights |
|---|---|---|---|---|---|
| `head/layer1` | 20 | 20 | 451.6 M | 26.9% | 38 KB |
| `xvector/block2` | 1416 | 624 | 366.2 M | 21.8% | 2687 KB |
| `xvector/block3` | 944 | 416 | 283.2 M | 16.9% | 2079 KB |
| `head/layer2` | 20 | 20 | 225.8 M | 13.5% | 38 KB |
| `xvector/block1` | 708 | 312 | 109.9 M | 6.5% | 803 KB |
| `xvector/transit2` | 5 | 5 | 78.1 M | 4.7% | 531 KB |
| `xvector/transit3` | 5 | 5 | 78.1 M | 4.7% | 533 KB |
| `xvector/tdnn` | 3 | 3 | 30.5 M | 1.8% | 201 KB |
| `head/conv2` | 2 | 2 | 27.5 M | 1.6% | 9 KB |
| `xvector/transit1` | 5 | 5 | 19.5 M | 1.2% | 137 KB |
| `head/conv1` | 2 | 2 | 6.9 M | 0.4% | 1 KB |
| `xvector/dense` | 8 | 6 | 0.2 M | 0.0% | 194 KB |
| `head` | 24 | 7 | 0.0 M | 0.0% | 0 KB |
| `xvector/stats` | 16 | 9 | 0.0 M | 0.0% | 0 KB |

## Fusion candidates

Maximal straight-line runs with no fan-out. Every node after the first in a run is a kernel launch and a round trip to LPDDR that a fused kernel removes.

- 224 fusable runs covering 1283 of 1438 runtime nodes
- collapsing them removes **1059 kernel launches** (74% of runtime nodes)

| x | intermediate traffic | kernel | pattern |
|---|---|---|---|
| 52 | 43.93 MB | `BNAffineReluQuantConvPack` | `BatchNormalization -> Relu -> QuantizeLinear -> QLinearConv -> DequantizeLinear -> Relu` |
| 52 | 10.64 MB | `CAMMaskGen` | `Add -> QuantizeLinear -> QLinearConv -> DequantizeLinear -> Relu -> QuantizeLinear -> QLinearConv -> DequantizeLinear -> Sigmoid` |
| 52 | 10.26 MB | `SegmentContextConsumerFusion` | `AveragePool -> Unsqueeze -> Expand -> Reshape -> Slice` |
| 1 | 6.75 MB | `ConvReluRequant` | `Transpose -> Unsqueeze -> QuantizeLinear -> QLinearConv -> DequantizeLinear -> Relu -> QuantizeLinear` |
| 2 | 6.55 MB | `ConvReluRequant` | `QuantizeLinear -> QLinearConv -> DequantizeLinear -> Relu -> QuantizeLinear -> QLinearConv -> DequantizeLinear` |
| 2 | 6.00 MB | `ConvReluRequant` | `QLinearConv -> DequantizeLinear -> Relu -> QuantizeLinear -> QLinearConv -> DequantizeLinear` |
| 1 | 3.00 MB | `RequantBoundary` | `Add -> Relu -> QuantizeLinear -> QLinearConv -> DequantizeLinear -> Relu -> Reshape -> QuantizeLinear -> QLinearConv -> DequantizeLinear -> Relu` |
| 2 | 2.95 MB | `DenseConcatBNReluQuantConvPack` | `Concat -> BatchNormalization -> Relu -> QuantizeLinear -> QLinearConv -> DequantizeLinear` |
| 1 | 2.91 MB | — | `Add -> Relu -> QuantizeLinear` |
| 1 | 2.26 MB | `DenseConcatBNReluQuantConvPack` | `Concat -> BatchNormalization -> Relu -> QuantizeLinear -> QLinearConv -> DequantizeLinear -> Relu` |
| 2 | 2.18 MB | — | `Add -> Relu` |
| 52 | 1.18 MB | `RequantBoundary` | `QuantizeLinear -> QLinearConv -> DequantizeLinear` |
| 1 | 0.59 MB | `StatsPoolingStd` | `Sub -> Mul -> ReduceMean -> Mul -> Div -> Sqrt` |
| 2 | 0.55 MB | — | `QLinearConv -> DequantizeLinear` |
| 1 | 0.01 MB | `RequantBoundary` | `Concat -> Unsqueeze -> QuantizeLinear -> QLinearConv -> DequantizeLinear -> Squeeze -> BatchNormalization` |

### What each kernel is allowed to do

**`BNAffineReluQuantConvPack`** — NOT a BN-into-conv-weight fold. A ReLU sits between the BN and the conv, and BN's own input is the dense Concat, so BN cannot be folded forward into the conv weights nor backward into the producers. What fuses is the input side: BN affine + ReLU + quantize become one pass that writes the int8 conv input directly, so neither the float BN output nor the float ReLU output is ever materialized.

**`CAMMaskGen`** — The whole CAM mask bottleneck (C -> C/2 -> C/4 -> sigmoid). Small channel counts, so kernel launch overhead dominates the arithmetic.

**`SegmentContextConsumerFusion`** — Segment-level context broadcast back onto the frame axis. This collapses to address arithmetic only if the consumer is fused in too -- the Add, and through it the CAM mask conv. Deleting Expand while still handing the result to a plain ORT tensor does not work: the consumer requires a materialized contiguous tensor, which is exactly the write we are trying to avoid.

**`ConvReluRequant`** — Dequantize -> ReLU -> requantize around a conv output. ReLU on an affine-quantized tensor is a clamp at the zero point, so this is a scale adjustment plus a clamp folded into the conv epilogue.

**`RequantBoundary`** — Pure quantization boundary. Once neighbouring kernels agree on a scale it disappears entirely.

**`DenseConcatBNReluQuantConvPack`** — Dense connection followed by the same packing epilogue as BNAffineReluQuantConvPack. The Concat itself is removable separately: preallocate the full channel block once and have each unit write its output into its own slice, so nothing is copied.

**`StatsPoolingStd`** — Standard-deviation branch of the statistics pooling layer.

## Tensor arena

| bucket | naive (one buffer per tensor) | arena (offline planner) | theoretical floor | overhead | buffers |
|---|---|---|---|---|---|
| 1 s | 47.12 MB | 1.91 MB | 1.91 MB | 0.0% | 1439 |
| 3 s | 137.82 MB | 5.82 MB | 5.82 MB | 0.0% | 1439 |
| 5 s | 228.53 MB | 9.73 MB | 9.73 MB | 0.0% | 1439 |
| 10 s | 452.74 MB | 19.49 MB | 19.49 MB | 0.0% | 1439 |

Largest live tensors at the peak step of the 3 s bucket:

- `/head/conv1/Conv_output_0` — 2980 KB
- `/head/Relu_output_0` — 2980 KB

## Shape verification

- 1 s: 3180 tensors checked, 0 mismatched
- 3 s: 3180 tensors checked, 0 mismatched
- 5 s: 3180 tensors checked, 0 mismatched
- 10 s: 3180 tensors checked, 0 mismatched
