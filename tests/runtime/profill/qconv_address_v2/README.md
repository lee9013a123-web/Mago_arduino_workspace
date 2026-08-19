# QConv address v2 tests

구현 시 다음 단독 테스트를 이 디렉터리에 둔다.

- 1x1, 3x3 interior, border, spatial tail의 pointer/NULL 결과 비교
- 기존 `campp_qconv_address_input_base()` 및 V4 tile plan과 전 case 비교
- storage-span 경계와 canary를 포함한 memory-safety 검사
- input bucket 98, 고정 feature 입력에서 operator replay retained tensor bitwise 검사
- address-v2 단독 latency와 V4 통합 latency를 분리한 benchmark

테스트 이름은 `test_qconv_address_v2_*.c` 또는 `test_qconv_address_v2_*.py`를
사용한다.
