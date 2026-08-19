# Microkernels

현재 `qlinear_conv/microkernels/qconv_mac_4x8.*` ABI와 검증 코드를 재사용하고
shape별 hot loop만 추가한다.

예정 파일:

```text
qconv_mac_1x1_8x8.h
qconv_mac_1x1_8x8_intrinsics.c
qconv_mac_3x3_interior_8x8.h
qconv_mac_3x3_interior_8x8_intrinsics.c
qconv_mac_tail.h
qconv_mac_tail_intrinsics.c
```

- 1x1: kernel-coordinate와 padding 분기 제거
- 3x3 interior: 9개 input pointer를 한 번 구성하고 tile 내에서 증가
- tail: spatial 1~7개와 output channel 1~7개를 별도 처리
- 미지원 dtype/layout/range: 기존 `mac_fixed`로 fallback

assembly는 intrinsics의 bitwise·latency·spill 결과가 나온 뒤에만 추가한다.

