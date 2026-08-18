# Candidate allocation

진단 결과가 생성된 뒤 다음 경로에 한 후보씩 구현한다.

```text
candidates/
├── qlinear_conv/            # QConv address/MAC/combined 후보
├── fused_quant_qconv/       # full scratch 대신 tile quantization 실험
├── bn_relu_quant/           # channel affine precompute/NEON 실험
└── dequantize_linear/       # contiguous/per-axis NEON 실험
```

아직 진단 결과가 없는 후보 소스는 만들지 않는다. 이 규칙은 측정 전에 병목을
단정하거나 production 코드와 중복 구현을 남기는 것을 방지한다.

`qlinear_conv/`의 후보는 production registry에 등록하지 않는다.
`campp_operator_microbench --qconv-candidate`가 target Operator에서만 후보
entry로 교체하므로 graph prelude와 다른 Operator는 기존 runtime을 사용한다.

```text
qlinear_conv/
├── qconv_address_fastpath.h/.c  # channel-packed 입력의 직접 spatial offset
├── qconv_mac_neon.h/.c          # O4I4를 4 output lane으로 누적하는 NEON MAC
└── qconv_candidate.h/.c         # address/mac/combined 조합과 기존 fallback
```

- `address`: 직접 offset + scalar MAC
- `mac`: 일반 Tensor offset + NEON MAC
- `combined`: 직접 offset + NEON MAC

지원하지 않는 layout이나 안전한 int32 부분합 범위를 벗어나는 shape는 기존
`campp_aarch64_qlinear_conv_o4i4()`로 fallback한다.
