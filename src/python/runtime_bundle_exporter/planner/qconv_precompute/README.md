# QConv parameter precompute

QConv v4의 오프라인 parameter 후보를 생성하는 Python planner 영역이다.

예정 모듈:

```text
multiplier_planner.py    # float scale에서 multiplier/shift 후보 생성
parameter_manifest.py   # 채널별 metadata와 checksum
```

초기 단계에서는 분석 결과만 만들고 bundle에는 기록하지 않는다. fixed-point 결과가
기존 float requant와 bitwise 동일하다는 검증 뒤에만 execution plan/weight blob
writer를 확장한다.

