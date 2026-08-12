# Runtime tests

CAM++ Reference Runtime과 runtime bundle exporter의 검증 영역이다.

| 디렉터리 | 역할 |
|---|---|
| `kernel_unit_tests/` | 작은 수동 Tensor로 개별 C kernel의 padding, stride, axis, broadcasting 검증 |
| `operator_replay_tests/` | ORT에서 저장한 실제 operator 입력을 C kernel에 재생 |
| `end_to_end_tests/` | 1, 3, 5, 10초 bucket의 block 경계와 최종 embedding 비교 |
| `memory_safety_tests/` | buffer 범위 초과, Arena offset·수명 충돌, 독립 buffer와 Arena의 bit-exact 동등성 검증 |

Python/C binary format 호환성과 exporter 결과 검증 테스트도 이 폴더의 최상위에 추가한다.
