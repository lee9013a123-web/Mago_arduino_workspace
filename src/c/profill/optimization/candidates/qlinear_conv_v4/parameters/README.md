# Parameters

output-channel block별 weight zero point, bias, float multiplier와 향후 검증된
fixed-point multiplier/shift를 읽기 전용 view로 제공한다.

예정 파일:

```text
qconv_v4_parameters.h
qconv_v4_parameters.c
```

첫 구현은 현재 bundle의 scale을 kernel 호출당 한 번 변환한다. 오프라인 metadata는
`src/python/runtime_bundle_exporter/planner/qconv_precompute/`에서 생성하며,
binary-format 확장 전에는 기존 bundle과 호환되는 runtime 계획만 사용한다.

