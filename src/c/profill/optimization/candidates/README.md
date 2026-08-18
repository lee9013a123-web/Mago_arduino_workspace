# Candidate allocation

진단 결과가 생성된 뒤 다음 경로에 한 후보씩 구현한다.

```text
candidates/
├── qlinear_conv/            # QConv address/MAC/combined 후보
├── fused_quant_qconv/       # 기존 fused quantization + QConv 후보 결합
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

`fused_quant_qconv/`는 quantization과 scratch layout을 다시 구현하지 않는다.
production의 공용 fused driver에 검증된 QConv candidate runner만 주입해,
일반 QConv에서 얻은 MAC/combined 개선이 fused Quant-QConv에도 그대로 적용되는지
분리 측정한다.

```text
fused_quant_qconv/
└── fused_quant_qconv_candidate.h/.c  # baseline/mac/combined dispatch adapter
```

- `baseline`: production fused Quant-QConv를 그대로 실행
- `mac`: 기존 full-tensor quantization 뒤 QConv MAC candidate 실행
- `combined`: 기존 full-tensor quantization 뒤 QConv combined candidate 실행

따라서 세 모드의 차이는 QConv runner뿐이다. input quantization, scratch 크기,
입출력 descriptor, requantization 의미는 production 경로와 동일하다.

`bn_relu_quant/` 후보 역시 target Operator에서만 교체한다.

```text
bn_relu_quant/
├── bn_iteration_fastpath.h/.c  # N/C/spatial offset을 직접 증가
├── bn_affine_fastpath.h/.c     # 채널별 scale/sqrt/bias 계산
├── bn_quant_neon.h/.c          # 4-lane ReLU/quantize 변환
└── bn_candidate.h/.c           # address/affine/quant/combined 조합
```

- `address`: 직접 input/output offset, 기존처럼 element마다 affine 계산
- `affine`: 채널별 multiplier/additive 사전 계산, 일반 Tensor 접근 유지
- `quant`: 일반 Tensor 접근과 element별 affine, 4-lane quantize
- `combined`: 채널별 affine + 직접 offset + channel-packed 4-lane NEON

후보는 최대 4096 channel을 실험 범위로 두며 범위를 벗어나거나 지원하지 않는
layout이면 `campp_fused_bn_relu_quant()`를 호출한다. production 승격 전에는
bundle format이나 runtime scratch plan을 바꾸지 않는다.
