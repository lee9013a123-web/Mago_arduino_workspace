# Planning

Tensor layout, convolution attribute, tile 범위, 직접 store 가능 여부와 parameter
주소를 kernel 실행 초기에 한 번 계산한다.

구현 파일:

```text
qconv_v4_execution_plan.h
qconv_v4_execution_plan.c
qconv_v4_tile_plan.h
qconv_v4_tile_plan.c
```

hot loop에는 division, attribute lookup, `isfinite`, generic stride 판정을 남기지
않는다. plan 생성에 실패하면 기존 `mac_fixed` 경로로 되돌아간다.
