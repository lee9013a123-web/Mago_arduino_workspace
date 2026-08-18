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

`dequantize_linear/`는 E7 channel-packed INT8/UINT8 입력을 저장 순서대로
순회한다. 미지원 rank·stride·per-axis 축은 reference kernel로 fallback한다.

```text
dequantize_linear/
├── dequant_layout_plan.h/.c       # N/spatial/channel 직접 offset 계획
├── dequant_scalar_fastpath.h/.c   # 직접 read, scalar 변환, FP32 store
├── dequant_neon.h/.c              # 16-lane INT8/UINT8→FP32 변환
└── dequant_candidate.h/.c         # 후보 mode와 reference fallback
```

- `address`: 저장 순서 직접 pointer, 기존 parameter read 유지
- `parameter`: generic Tensor 접근, scale/zero point hoist
- `scalar_combined`: 직접 pointer + parameter hoist + scalar 변환
- `neon_combined`: combined 구조에 16-lane NEON 변환 추가

모든 scale은 fast loop 진입 전에 양수·finite인지 확인한다. scalar scale과
channel-axis per-axis scale을 지원하고 나머지는 production 의미를 유지하기 위해
`campp_reference_dequantize_linear()`로 되돌아간다.
