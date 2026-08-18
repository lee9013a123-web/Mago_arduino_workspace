# Candidate allocation

진단 결과가 생성된 뒤 다음 경로에 한 후보씩 구현한다.

```text
candidates/
├── common/                  # 일반/fused QConv 공통 O4I4 microkernel
├── qlinear_conv/            # 3x3, 1x1 shape dispatch 실험
├── fused_quant_qconv/       # full scratch 대신 tile quantization 실험
├── bn_relu_quant/           # channel affine precompute/NEON 실험
└── dequantize_linear/       # contiguous/per-axis NEON 실험
```

아직 진단 결과가 없는 후보 소스는 만들지 않는다. 이 규칙은 측정 전에 병목을
단정하거나 production 코드와 중복 구현을 남기는 것을 방지한다.
