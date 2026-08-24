# Requantization

현재 production 4-lane `campp_aarch64_qconv_requantize_store4()` 의미를 유지하면서
변환 폭을 8 lane으로 확대한다.

구현 파일:

```text
qconv_requant_neon8.h
qconv_requant_neon8.c
```

`qconv_requant_fixedpoint.*`는 bitwise 증명 뒤에 추가할 예약 경로이며 현재 v4
dispatch에는 포함하지 않는다.

검증 항목:

- INT32 경계, 양·음 accumulator
- UINT8/INT8 output zero point
- ties-to-even rounding
- saturation 경계
- output-channel 1~7 tail

fixed-point 경로는 float reference와 모든 retained tensor가 bitwise 동일할 때만
dispatch 후보가 된다.
