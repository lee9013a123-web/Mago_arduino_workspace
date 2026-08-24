# Dispatch

Operator당 한 번 shape와 layout을 분류하고 아래 구현을 선택한다.

구현 파일:

```text
qconv_v4_dispatch.h
qconv_v4_dispatch.c
```

예정 분류:

- 1x1 dense interior
- 3x3 interior
- spatial tail
- output-channel tail
- 기존 `mac_fixed` fallback
- production generic fallback

함수와 타입은 각각 `campp_qconv_v4_`, `CamppQconvV4` prefix를 사용한다.
