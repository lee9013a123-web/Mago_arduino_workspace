# QConv v4 tests

구현 테스트:

```text
test_qconv_v4_primitives.c
../test_qconv_candidate.c      # v4 mode까지 production output과 byte 비교
```

C test는 production 또는 cache-packed reference와 output byte를 비교한다. 최종
candidate test는 일반 QConv와 fused Quant-QConv 양쪽에 같은 runner를 주입하고,
고정 입력 3개의 retained tensor 전체를 stock runtime dump와 비교한다.
