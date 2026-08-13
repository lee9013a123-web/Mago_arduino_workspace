# CAM++ Reference Runtime

Arduino UNO Q의 QRB2210 Linux 영역에서 CAM++ 정적 실행 계획을 수행하는 C Runtime 구현 영역이다.

## 디렉터리 책임

| 디렉터리 | 역할 |
|---|---|
| `include/campp_runtime/` | 공개 API와 Tensor, Operator, binary descriptor 선언 |
| `internal/` | Runtime 내부 구조체. 로드된 모델, 실행 상태, kernel view와 registry |
| `model_loading/` | `weights.bin`과 `plan_*.bin` 로딩 및 검증 |
| `execution/` | execution table 순회, Tensor 연결, kernel dispatch |
| `memory_management/` | 초기 Tensor별 독립 버퍼와 이후 정적 Tensor Arena 관리 |
| `backends/cpu_reference/` | SIMD 없는 정확도 기준 C kernel 구현 |
| `backends/cpu_aarch64/` | O4I4 QLinearConv spatial tile과 AArch64 NEON kernel |
| `diagnostics/` | 중간 Tensor dump, 실행 trace, 메모리 기록 |
| `platform_linux/` | 정렬 할당과 Linux 종속 메모리·CPU 기능 처리 |
| `command_line/` | 터미널에서 Reference Runtime을 실행하는 프로그램 |

## 최초 구현 범위

1. compiled model loader와 validator
2. Tensor registry와 순차 graph executor
3. Tensor별 독립 buffer를 사용하는 reference storage
4. 현재 정적 그래프에 존재하는 CPU Reference operator
5. 중간 Tensor dump와 ORT 결과 비교 지원
6. exporter의 정적 offset을 사용하는 단일 Tensor Arena

Packed plan은 kernel ID 1을 사용한다. AArch64 backend는 QLinearConv에
`[group][O4][kernel][I4][output lane][input lane]` weight와 spatial tile 8을
적용하고, 나머지 opcode는 stride-aware reference kernel로 fallback한다.
비-AArch64 build도 같은 packed layout을 scalar로 실행하므로 개발 PC에서
bitwise 검증할 수 있다. Runtime weight packing은 수행하지 않는다.

`command_line/campp_runtime_benchmark.c`는 진단 callback 없이 같은 context를
반복 실행하며 raw timing, 초기화 시간, `/proc/self/status`의 RSS와 embedding을
JSON/float32로 남긴다. `scripts/3_runtime/09_benchmark_runtime.py`가 ORT와
같은 입력 및 프로세스 수명 규칙으로 이 실행 파일을 호출한다.
