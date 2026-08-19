# QConv v4 tests

예정 테스트:

```text
test_qconv_v4_plan.c
test_qconv_requant_neon8.c
test_qconv_mac_1x1_8x8.c
test_qconv_mac_3x3_interior_8x8.c
test_qconv_mac_tail.c
test_qconv_v4_candidate.c
test_qconv_v4_retained_tensors.py
```

C test는 production 또는 cache-packed reference와 output byte를 비교한다. 최종
candidate test는 일반 QConv와 fused Quant-QConv 양쪽에 같은 runner를 주입하고,
고정 입력 3개의 retained tensor 전체를 stock runtime dump와 비교한다.

