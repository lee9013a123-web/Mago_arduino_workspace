# QConv parameter precompute

QConv v4의 오프라인 parameter 후보를 생성하는 Python planner 영역이다.

예정 모듈:

```text
multiplier_planner.py    # float scale에서 multiplier/shift 후보 생성
parameter_manifest.py   # 채널별 metadata와 checksum
qconv_weight_zero_points.py  # weight zero-point fast-path 판정
qconv_weight_sums.py         # output-channel별 int8 weight sum
qconv_corrected_bias.py      # input zero-point를 반영한 corrected bias
qconv_v5_pack_plan.py        # v5 packed-weight manifest 후보
```

초기 단계에서는 분석 결과만 만들고 bundle에는 기록하지 않는다. fixed-point 결과가
기존 float requant와 bitwise 동일하다는 검증 뒤에만 execution plan/weight blob
writer를 확장한다.

v5 helper는 아직 bundle format을 바꾸지 않는다. Python planner에서 같은 값을
재현 가능하게 계산하고, C 후보가 bitwise gate를 통과한 뒤 writer/loader를
동기화한다.
