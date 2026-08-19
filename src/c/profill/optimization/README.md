# E7 optimization diagnostics

이 디렉터리는 production kernel을 바꾸기 전에 상위 kernel family의 내부
병목을 확인하는 격리된 진단 환경이다.

- `diagnostics/`: 단계별 wall time과 target-only Linux PMU counter 수집
- `command_line/`: 단일 Operator 진단과 입력당 1회 graph 순회의 family 측정
- `candidates/`: 진단 결과로 채택된 실험 구현만 추가하는 staging 영역

후보 구현은 기본 AArch64 registry에 등록하지 않는다. 세 입력의 output hash,
retained tensor bitwise 검증, Quick E2E 성능을 모두 통과한 구현만
`src/c/runtime/backends/cpu_aarch64/`로 이동한다. production runtime이 이
디렉터리에 의존하게 만들지 않는다.

QConv 일반 경로와 fused Quant-QConv는 동일한 O4I4 core를 공유한다. QConv
후보는 `candidates/qlinear_conv/`에 모으고 address, MAC, combined와 고정
S4×O8 microkernel 모드로 각각 측정한다. `mac_fixed`는 compiler register
allocation을 비교하고 `mac_asm`은 fixed intrinsics에도 spill이 남을 때만
검증한다. fused 후보는 production의 validation/scratch driver를 그대로
사용하고 QConv runner와 input quantizer를 독립적으로 주입한다. `mac_fixed`는
공통 QConv 효과, `quant_neon`은 fused input pass 효과, `combined_fixed`는 두
효과를 함께 측정한다. 따라서 별도 MAC 구현 없이 같은 개선을 공유한다. 이후
BN-ReLU-Quant, DequantizeLinear를 분리한다.

전체 fused family 비교는 `campp_fused_qconv_family_bench`가 graph를 입력당
한 번만 실행한다. 각 target에서 activation 입력을 잠시 복사해 후보들을 같은
값으로 측정하고 baseline을 마지막에 실행해 그 출력을 graph 진행에 사용한다.
따라서 Operator별 prelude 재실행은 측정시간에 포함되지 않는다.

BN 후보도 production registry와 분리한다. `address`, `affine`, `quant`는
각 원인의 독립 효과를 확인하고 `combined`는 channel-packed 입력에서 채널
4개를 함께 처리한다. 지원하지 않는 shape, stride, rounding mode는 기존
`campp_fused_bn_relu_quant()`로 되돌아간다.

DequantizeLinear 후보 역시 production registry에 등록하지 않는다. 기존
channel-packed Tensor view를 `N/spatial/channel` 순서로 직접 순회하고 scalar
parameter를 hoist한 뒤, 최종 후보에서 16개 channel을 NEON으로 변환한다.
지원하지 않는 layout과 axis는 reference kernel로 fallback한다.

나머지 10개 kernel 후보는 `candidates/common/`의 packed iteration, NEON
elementwise, reduction, block-copy, Sigmoid LUT를 공유한다. 후보 선택은
`--remaining-candidate baseline|optimized`로 격리하며, 미지원 signature는 기존
kernel로 돌아간다. 승격 전에는 개별 output hash뿐 아니라 retained tensor 전체와
Quick E2E를 다시 검증한다.

`perf_sample_window.c`는 외부 Linux `perf record`를 target kernel 호출 동안만
활성화한다. sampling 전용 `campp_operator_hotspot`은 runtime kernel의 stage
probe를 컴파일하지 않아 소스 라인 표본에 계측 clock 호출이 섞이지 않는다.
