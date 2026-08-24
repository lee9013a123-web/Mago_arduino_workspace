# QConv v4 experiments

QConv v4 후보를 기존 final suite와 분리해 QRB2210에서 측정하는 script 영역이다.

실행 순서:

```text
01_build_qconv_v4.sh
02_profile_qconv_v4_stages.py
03_benchmark_qconv_v4.py
04_compare_qconv_v4.py
```

측정 case는 기존 고정 입력 3개와 대표 3x3/1x1 Operator를 그대로 사용한다.
현재 구현 mode는 `baseline`, `mac_fixed`, `v4`다. v4는 1x1/3x3 interior 주소
계획, fixed 8-output MAC, 8-lane requant/store를 묶고 spatial/channel tail은
MAC-v2로 fallback한다.

```bash
bash scripts/4_profill/optimization/qconv_v4/01_build_qconv_v4.sh
python3 scripts/4_profill/optimization/qconv_v4/03_benchmark_qconv_v4.py \
  --preflight-only
python3 scripts/4_profill/optimization/qconv_v4/03_benchmark_qconv_v4.py --force
```

결과 경로:

```text
runs/profiling/e7_98/optimization/qconv_v4/<mode>/
results/profiling/e7_98/optimization/qconv_v4/<mode>.json
```

각 후보는 output bitwise, shape별 1% 초과 회귀 없음, CV 3% 이하를 통과해야 다음
조합에 포함한다.
