# Planning

`qconv_v5_execution_plan.*`는 기존 검증된 V4 execution plan을 한 번 생성한 뒤
address-neutral geometry와 row/interior plan으로 변환한다. Hot loop와
microkernel은 Tensor descriptor나 convolution attribute를 다시 검사하지 않는다.

`qconv_v5_validated_raw_plan.*`는 output block별 MAC invariant를 확정해
microkernel의 반복 `is_supported` 검사를 제거한다.

Planning hoists validation out of the MAC tile loop.  The first target is a
validated raw plan that records whether a QConv block may enter a no-guard
microkernel path.
