# Channel-packed store

layout 검사가 끝난 output에 8-byte 또는 16-byte 연속 store를 수행한다.

예정 파일:

```text
qconv_store_channel_packed.h
qconv_store_channel_packed.c
```

- full output block: 단일 연속 store
- non-aliased 마지막 tail: 소유한 padding 범위까지 연속 store
- aliased tail: 유효 byte만 저장
- generic stride: production store fallback

offset·capacity·alias 검사는 `planning/`에서 끝내고 store 함수에는 검증된 범위만
전달한다.

