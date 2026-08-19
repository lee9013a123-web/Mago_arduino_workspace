# Planning

`CamppQconvAddressGeometry`를 검증하고 다음 값을 hot loop 밖에서 계산한다.

- output 및 kernel pointer 증가량
- 3x3 interior의 row/column 시작과 끝
- output spatial count와 kernel element count
- 첫 point와 마지막 point를 이용한 storage 범위 증명
- `/`, `%` 없이 left/interior/right work item을 만드는 row cursor

이 계층은 `CamppRuntimeModel`, `CamppOperatorDescriptor`, V4 execution plan 또는
MAC header를 include하지 않는다.
