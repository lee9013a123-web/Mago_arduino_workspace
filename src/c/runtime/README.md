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

AArch64 NEON과 fused kernel은 Reference Runtime 및 Tensor Arena 결과를 기준으로
다음 단계에서 추가한다.
