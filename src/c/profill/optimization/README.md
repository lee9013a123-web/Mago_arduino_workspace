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

QConv 일반 경로와 fused Quant-QConv는 동일한 O4I4 core를 공유한다. 하지만 첫
후보는 MAC sampling 결과가 두 대표 shape에서 안정적으로 MAC 우세일 때에만
`candidates/common/`의 공통 microkernel으로 정한다. 주소·load·제어가
우세하면 QConv address fast path를 먼저 만들고, shape별 승자가 다르면 3x3과
1x1 후보를 분리한다. 이후 fused input tile quantization, BN-ReLU-Quant,
DequantizeLinear 순으로 분리한다.

`perf_sample_window.c`는 외부 Linux `perf record`를 target kernel 호출 동안만
활성화한다. sampling 전용 `campp_operator_hotspot`은 runtime kernel의 stage
probe를 컴파일하지 않아 소스 라인 표본에 계측 clock 호출이 섞이지 않는다.
