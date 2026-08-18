# E7 operator fusion 검증

E7은 `runs/runtime/kernel_optimization/cache_packed/bundle`을 baseline으로 사용하고, fusion 이후
operator table, Tensor ID, alias lifetime, Arena offset과 kernel ID를 모두 다시
생성한다.

독립적으로 제어하는 fusion family는 다음과 같다.

- `FUSION_CONV_BIAS_ACT`
- `FUSION_BN_RELU_QUANT`
- `FUSION_POOL_CAM`
- `FUSION_QDQ_ELEMENTWISE`
- `FUSION_STATS_POOLING`

현재 hybrid graph에서 bit-exact하게 적용된 pattern은 BN-ReLU-Quantize,
Quantize-QLinearConv input packing, Dequantize-ReLU-Quantize, CAM mask의
Dequantize-Sigmoid-Mul, statistics mean/std reduction이다. 별도 QDQ elementwise
직선 chain은 현재 graph에 없어 적용 수가 0이며 kernel과 planner 경로만 제공한다.

평가는 고정된 98-frame 입력 세 개만 사용한다.

```bash
python3 scripts/3_runtime/14_export_e7_fused_bundle.py --force
python3 scripts/3_runtime/15_validate_e7_fusions.py \
  --runtime-binary build/campp_reference_dump
```

정확도 결과는 `e7_validation.json`, 세부 graph/memory 계획은
`runs/runtime/kernel_optimization/e7/bundle/fusion_plans/`에 기록된다.
