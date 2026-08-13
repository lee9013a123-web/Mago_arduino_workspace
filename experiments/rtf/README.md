# RTF 측정 루프 (짧은 음성)

C Runtime의 추론 시간을 줄여나가기 위한 반복 측정 실험이다. **정확도는 보지
않는다** — 그 판정은 `scripts/3_runtime/05_compare_runtime_outputs.py`가 한다.
최적화 후에는 RTF와 정확도를 각각 따로 확인해야 한다.

```
experiments/rtf/
├── README.md          이 문서 (baseline 조건)
├── measure_rtf.py     측정 스크립트
└── result/
    ├── baseline.json                        현재 baseline (항상 최신 기준선)
    ├── 20260813T033426Z__baseline.json      -O3 baseline 원본
    └── 20260813T032455Z__O2-reference.json  같은 커밋 -O2 참고 측정
```

## Baseline

**RTF p50 = 35.490** (2026-08-13, commit `56f5fc66`)

| 지표 | 값 |
|---|---|
| RTF p50 / mean / min | 35.490 / 35.483 / 35.433 |
| Latency p50 | 35,490 ms (1초 음성) |
| 표본 spread | 0.26% (9 samples) |
| Peak RSS | 12.24 MB |
| Arena | 2,007,040 B |
| model load | 33.1 ms (RTF에 미포함) |
| context create | 12.4 ms (RTF에 미포함) |

화자별 RTF p50: `0000` 35.495 / `0005` 35.470 / `0006` 35.442.

실시간의 **35배**다. 참고로 같은 입력에서 ONNX Runtime 1.27.0은 RTF p50
0.187(latency 186.6 ms)이므로 약 **190배** 차이가 난다. cpu_reference backend는
정확도 검증용 순수 C 구현이고 NEON·타일링·멀티스레드가 전혀 없다.

## 측정 조건

측정값을 비교하려면 아래 조건이 모두 같아야 한다.

**하드웨어 / OS**
- Arduino UnoQ (`arduino,imola`), aarch64, 논리 코어 4개
- kernel 7.0.0-g122c2c22d838
- CPU governor `schedutil` (4코어 전부)
- **CPU 0에 고정** (`os.sched_setaffinity`) — 스크립트가 자동으로 한다
- 측정 시작 온도 38.4°C

**빌드**
- gcc 14.2.0
- `CFLAGS="-std=c11 -O3 -DNDEBUG -Wall -Wextra"`
- `-O2`로 빌드하면 RTF가 39.146으로 **약 10% 느려진다**
  (`result/20260813T032455Z__O2-reference.json`). 빌드 플래그를 바꾸면 비교가
  깨지므로, 커널을 고치는 실험에서는 반드시 위 플래그를 쓴다.

```bash
CFLAGS="-std=c11 -O3 -DNDEBUG -Wall -Wextra" bash scripts/3_runtime/03_build_reference_runtime.sh
```

**런타임**
- `build/campp_runtime_benchmark`, backend `campp-c-reference`
- Bundle: `runs/runtime/tensor_arena/bundle` (Tensor Arena 배치)
- `--threads 1` (effective_threads 1, single-thread)
- Operator 1,438개, weights 7,453,696 B

**입력**
- bucket **98 frame = 1.0초** (짧은 음성)
- payload: `benchmarks/campplus/features/multi__speaker_{0000,0005,0006}__98.f32`
- manifest: `benchmarks/campplus/manifests/runtime_features.json`
- 실데이터 FBank다. `data/multi_speaker`의 화자별 3개 클립을 파일명 순으로
  이어붙이고 offset 0에서 1초를 잘라 만들었다. 생성 방법과 FBank 파라미터는
  `scripts/dataset_make/build_runtime_features.py`와 manifest의 `preprocessing`에
  기록되어 있다.

**프로토콜**
- 음성 3개 × (warmup 1 + repeat 3) = 유효 표본 9개
- RTF = `inference_ms / (audio_seconds × 1000)`
- **측정 범위는 추론 루프뿐이다.** WAV 읽기와 FBank는 payload를 미리 만들어
  제외했고, model load와 context create는 `lifecycle_ms`에 기록만 하고 RTF에는
  넣지 않는다.
- C Runtime은 매우 결정적이라(표본 spread 0.26%) repeat를 늘릴 실익이 적다.
  1회 추론이 35초라 repeat 3이면 실행에 약 8분 걸린다.

## 사용법

```bash
python3 experiments/rtf/measure_rtf.py --label baseline
```

`result/<UTC>__<label>.json`에 저장하고, `result/baseline.json`이 있으면 delta를
함께 출력한다.

```
RTF p50 32.100   mean 32.140   min 32.010   (9 samples, spread 0.31%)
baseline 'baseline' RTF p50 35.490 → 32.100  (-9.55%, 1.11x)
```

새 기준선으로 삼으려면:

```bash
python3 experiments/rtf/measure_rtf.py --label neon-qconv --save-as-baseline
```

주요 옵션 — `--bucket`(기본 98), `--inputs`(3), `--warmup`(1), `--repeat`(3),
`--threads`(1), `--cpu`(0), `--baseline <path>`.

더 긴 음성은 `--bucket 298|498|998`로 볼 수 있지만 시간이 frame 수에 비례해
늘어난다. 998은 1회 추론이 약 6분이라 기본 프로토콜로도 1시간이 넘는다.

## 주의

- 결과 JSON에 `git_commit`과 `git_dirty`가 들어간다. `git_dirty: true`면 커밋되지
  않은 변경이 섞인 측정이므로 기준선으로 쓰지 않는 편이 좋다. 현재 baseline은
  `git_dirty: true` 상태에서 측정됐다 (이 실험 폴더 자체가 미커밋이었다).
- 측정 중 보드에서 다른 작업을 돌리면 CPU 0 경합으로 값이 흔들린다.
- RTF가 좋아져도 정확도가 유지되는지는 별도로 확인해야 한다. 특히 부동소수점
  연산 순서를 바꾸는 최적화는 QuantizeLinear의 반올림 경계를 뒤집을 수 있다
  (`normalization_operators.c`의 BatchNorm affine 재작성 사례 참고).
