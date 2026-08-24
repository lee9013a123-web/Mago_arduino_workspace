# ONNX Runtime baseline (native C API vs Python)

정적 ONNX 모델을 ORT로 돌렸을 때의 latency / RTF / RSS 기준선이다. C Runtime
최적화 목표치를 정하기 위한 비교 대상이고, 정확도는 보지 않는다.

같은 모델·같은 payload·같은 스레드 설정으로 두 backend가 **같은 추론**을 한다.
차이는 프로세스에 Python 인터프리터와 numpy가 있느냐뿐이다.

```
experiments/baseline_ort_c/
├── README.md
├── ort_benchmark.c    native ORT 러너 (C API, Python 없음)
├── build.sh
├── measure.py         4 bucket × 2 backend 측정 드라이버
└── result/
    └── 20260813T054146Z__baseline.json
```

## Baseline

2026-08-13, commit `56f5fc66`, ORT 1.27.0, 화자 3명 × (warmup 5 + repeat 20).

| bucket | 음성 | native p50 | native RTF | python p50 | python RTF | native RSS | python RSS |
|---|---|---|---|---|---|---|---|
| 98 | 1.0s | **185.97 ms** | **0.186** | 186.50 ms | 0.187 | 67.93 MB | 100.45 MB |
| 298 | 3.0s | **522.44 ms** | **0.174** | 524.33 ms | 0.175 | 97.35 MB | 129.61 MB |
| 498 | 5.0s | **872.04 ms** | **0.174** | 865.40 ms | 0.173 | 126.01 MB | 158.40 MB |
| 998 | 10.0s | **1833.05 ms** | **0.183** | 1836.95 ms | 0.184 | 197.67 MB | 230.00 MB |

**Latency는 두 backend가 사실상 같다** (차이 ≤ 0.8%, 부호도 일정하지 않다).
Python `session.run()` 호출 오버헤드는 수백 ms 추론에 묻힌다. Python을 걷어내도
**추론 시간은 빨라지지 않는다.**

세션 생성은 native가 빠르다: 98에서 2,027 ms vs 1,934–3,035 ms (측정 편차 큼).

### 정확도

native와 Python의 embedding이 **비트 단위로 동일**하다 (98, max_abs 0.0,
cosine 1.000000000000). C API 경로가 Python 경로와 같은 계산을 한다는 확인이다.

## RSS를 어떻게 읽어야 하는가

측정값은 `/proc/self/status`의 **VmHWM(프로세스 peak RSS) 절대값**이다. 시스템
초기 RAM을 비우거나 page cache를 drop하지 않았고, 프로세스 자체의 baseline도
빼지 않았다. 그래서 위 표의 RSS에는 **각 런타임의 시작 시점 메모리가 그대로
포함되어 있다.**

프로세스 시작 시점(추론 전) RSS:

| | VmHWM |
|---|---|
| native (libonnxruntime.so 링크만 된 상태) | 7.46 MB |
| `python3 -c "pass"` | 9.17 MB |
| `python3 -c "import numpy"` | 25.37 MB |
| `python3 -c "import numpy, onnxruntime"` | **40.19 MB** |

이 baseline을 빼면 **ORT 자체 사용량은 두 backend가 같다**:

| bucket | native − 7.46 | python − 40.19 | 차이 |
|---|---|---|---|
| 98 | 60.47 MB | 60.26 MB | −0.22 MB |
| 298 | 89.89 MB | 89.42 MB | −0.47 MB |
| 498 | 118.55 MB | 118.21 MB | −0.34 MB |
| 998 | 190.21 MB | 189.81 MB | −0.40 MB |

**결론: Python 제거로 아끼는 32 MB는 전부 인터프리터 + numpy + pybind의 고정
바닥값이다.** ORT 쪽 메모리는 1 MB도 줄지 않는다. 4개 bucket에서 절감폭이
32.26–32.51 MB로 거의 일정한 것이 그 증거이고, 위 40.19 − 7.46 = 32.73 MB와도
일치한다.

즉 "Python 병목 제거"의 효과는 **고정 32 MB 절감이지 추론 성능 개선이 아니다.**

### RSS 측정의 한계

- page cache를 drop하지 않았다 (`drop_caches`는 sudo 비밀번호가 필요하다).
  RSS에는 큰 영향이 없지만 session create 시간에는 영향이 있다.
- 측정 시 시스템은 여유 상태였다 (free 809 MB, available 2,197 MB, swap 사용
  92 MB, load average 0.5–0.8). 측정 프로세스가 swap으로 밀린 흔적은 없다.
- VmHWM은 프로세스별 값이라 다른 프로세스의 메모리가 섞이지는 않는다. 다만
  공유 라이브러리의 resident 페이지가 포함되므로 "ORT가 요구하는 순수 메모리"와
  같지 않다.
- 더 정확한 값이 필요하면 `ort_benchmark.c`가 기록하는 **ORT allocator 통계**
  (`InUse`, `MaxInUse`, `TotalAllocated`, `NumArenaExtensions`)를 쓴다. 추론
  전후로 스냅샷을 떠서 `inferences[]`에 남긴다. RSS와 달리 ORT arena가 실제로
  쥐고 있는 바이트다.

## 측정 조건

**하드웨어 / OS** — `experiments/rtf/README.md`와 동일
- Arduino UnoQ (`arduino,imola`), aarch64, 4코어, governor `schedutil`
- **CPU 0 고정** (`os.sched_setaffinity`, 스크립트가 자동)

**모델** — `results/static/campp_static_{98,298,498,998}.onnx`
- 이미 graph 정리된 정적 shape 모델. 98은 노드 1,438개 / initializer 2,280개 /
  opset 11, 입력 `feature [1,98,80]`, 출력 `embedding [1,192]`
- canonical `models/source/campplus_int8_static_qop.onnx`는 노드 3,180개에
  동적 shape(`[batch_size, frame_num, 80]`)다. **09 벤치마크는 아직 canonical을
  쓴다** — 이 baseline과 직접 비교할 수 없다
- ORT 그래프 최적화는 `ORT_ENABLE_ALL`(기본), `ORT_SEQUENTIAL`, intra/inter op
  스레드 1

**입력** — `benchmarks/campplus/features/multi__speaker_{0000,0005,0006}__<N>.f32`
실데이터 FBank다. 생성 방법은 `scripts/dataset_make/README.md` 참고.

**프로토콜** — 화자 3명 × (warmup 5 + repeat 20) = bucket당 유효 표본 60개.
warm 추론 루프만 측정하고 session create는 따로 기록한다.

## 빌드

C API 헤더가 보드에 없어서 `.so`와 같은 버전(v1.27.0)을 vendoring 했다.

```
third_party/onnxruntime/include/onnxruntime_c_api.h     GitHub v1.27.0에서 받음
third_party/onnxruntime/include/onnxruntime_ep_c_api.h  위 헤더가 include 함
third_party/onnxruntime/lib/libonnxruntime.so.1         → venv wheel의 .so 심볼릭 링크
```

wheel에는 SONAME(`libonnxruntime.so.1`)에 해당하는 링크가 없어서 그대로 링크하면
실행 시 로더가 찾지 못한다. venv를 건드리지 않기 위해 저장소 안에 링크를 만들고
rpath를 그쪽으로 잡았다. 헤더의 `ORT_API_VERSION`(27)과 `.so` 버전이 어긋나면
`GetApi()`가 NULL을 준다.

```bash
bash experiments/baseline_ort_c/build.sh
```

## 사용법

```bash
python3 experiments/baseline_ort_c/measure.py --label baseline
python3 experiments/baseline_ort_c/measure.py --buckets 98 --native-only
```

주요 옵션 — `--buckets`(98 298 498 998) `--inputs`(3) `--warmup`(5) `--repeat`(20)
`--threads`(1) `--cpu`(0) `--graph-opt`(all) `--native-only`.

native 바이너리 단독 실행:

```bash
taskset -c 0 build/campp_ort_benchmark \
  --model results/static/campp_static_98.onnx \
  --input benchmarks/campplus/features/multi__speaker_0000__98.f32 \
  --audio-seconds 1.0 --warmup 5 --repeat 20 --threads 1
```

## C Runtime과의 거리

`experiments/rtf/`의 baseline(1초 음성, RTF 35.490)과 비교하면:

| | RTF (1초) | Peak RSS |
|---|---|---|
| ORT native | 0.186 | 67.93 MB |
| C Runtime (cpu_reference + arena) | 35.490 | **12.24 MB** |

C Runtime이 **191배 느리지만 메모리는 5.5배 적다**. cpu_reference는 NEON·타일링·
멀티스레드가 없는 순수 C 구현이라 속도 격차는 예상된 것이고, arena 덕에 activation
이 2 MB로 고정된다.

## 남은 일

- `09_benchmark_runtime.py`가 canonical ONNX를 쓴다. 이 baseline과 맞추려면
  `paths.canonical_model`을 bucket별 `results/static/campp_static_<N>.onnx`로
  바꾸는 구조 변경이 필요하다 (현재는 모델 경로가 하나다).
- `scripts/1_benchmark/benchmark_onnx.py`는 그대로 재사용 중이다. Python 측
  지표는 이 스크립트가 내는 값을 `measure.py`가 파싱한다.
