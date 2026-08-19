# Fused BN/ReLU/Quant v2 candidate layout

`bn_relu_quant_v2/` is a staging implementation that keeps the existing
`bn_relu_quant/combined` kernel as the benchmark baseline. It is not selected
by the final candidate suite until retained-tensor bitwise validation and the
board performance gate pass.

```text
bn_relu_quant_v2/
├── planning/       # invocation validation, layout and direct-pointer plan
├── parameters/     # aligned affine or quant-prescaled coefficient blocks
├── dispatch/       # exact16, spatial2 and prescaled path selection
└── microkernels/   # 16-channel NEON affine/ReLU/quantize/packed-store kernels
```

`planning` is the only hot-path module that interprets tensor descriptors.
Microkernels receive contiguous pointers, channel counts and prepared POD
coefficients. Unsupported or overlapping layouts use the production fused BN
fallback.

Candidate modes:

- `v2_exact16`: exact operation order, one spatial position by 16 channels.
- `v2_spatial2`: exact operation order, two spatial positions sharing affine
  coefficient loads.
- `v2_prescaled`: folds quant scale into affine coefficients; experimental
  until output hashes and retained tensors are bitwise identical.
