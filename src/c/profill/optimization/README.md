# E7 optimization diagnostics

이 디렉터리는 production kernel을 바꾸기 전에 상위 kernel family의 내부
병목을 확인하는 격리된 진단 환경이다.

- `diagnostics/`: 단계별 wall time과 target-only Linux PMU counter 수집
- `command_line/`: 실제 E7 중간 Tensor를 만든 뒤 Operator 하나만 반복 실행
- `candidates/`: 진단 결과로 채택된 실험 구현만 추가하는 staging 영역

후보 구현은 기본 AArch64 registry에 등록하지 않는다. 세 입력의 output hash,
retained tensor bitwise 검증, Quick E2E 성능을 모두 통과한 구현만
`src/c/runtime/backends/cpu_aarch64/`로 이동한다. production runtime이 이
디렉터리에 의존하게 만들지 않는다.

QConv 일반 경로와 fused Quant-QConv는 동일한 O4I4 core를 공유한다. QConv
후보는 `candidates/qlinear_conv/`에 모으고 address, MAC, combined 모드로
각각 측정한다. fused 후보는 production의 quantization/scratch driver를 그대로
사용하고 QConv runner만 주입한다. 따라서 별도 MAC 구현 없이 같은 개선을
공유하며, fused 측정에서 남는 input quantization 비중은 후속 tile quantization
후보의 근거로 사용한다. 이후 BN-ReLU-Quant, DequantizeLinear를 분리한다.

BN 후보도 production registry와 분리한다. `address`, `affine`, `quant`는
각 원인의 독립 효과를 확인하고 `combined`는 channel-packed 입력에서 채널
4개를 함께 처리한다. 지원하지 않는 shape, stride, rounding mode는 기존
`campp_fused_bn_relu_quant()`로 되돌아간다.

`perf_sample_window.c`는 외부 Linux `perf record`를 target kernel 호출 동안만
활성화한다. sampling 전용 `campp_operator_hotspot`은 runtime kernel의 stage
probe를 컴파일하지 않아 소스 라인 표본에 계측 clock 호출이 섞이지 않는다.
