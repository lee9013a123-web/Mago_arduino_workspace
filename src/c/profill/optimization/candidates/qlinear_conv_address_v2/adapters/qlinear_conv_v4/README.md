# V4 adapter

이 디렉터리만 V4 execution-plan header와 address-v2 API를 함께 include한다.

책임:

- `CamppQconvV4ExecutionPlan`을 `CamppQconvAddressGeometry`로 한 번 변환
- address 경로 선택 결과를 V4 dispatch에 전달
- transition 동안 `CamppQconvAddressTile`을 기존 V4 MAC tile ABI로 변환
- unsupported case를 기존 V4/mac_fixed generic 경로로 fallback

MAC multiply와 dispatch 최적화가 완료되기 전에는 기존 V4 파일을 수정하지
않고 adapter 구현과 단독 테스트까지만 진행한다.
