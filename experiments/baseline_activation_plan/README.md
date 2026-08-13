# Native ORT 4-bucket RAM/RTF baseline

이 실험은 기존 `experiments/baseline_ort_c/ort_benchmark.c`를 사용하여 CAM++
정적 모델 4개를 같은 조건으로 측정한다. Python은 fresh native child process를
실행하고 JSON을 분석할 뿐이므로 모든 RSS/PSS 값에서 제외된다.

## 고정 조건

- 모델: `results/static/campp_static_{98,298,498,998}.onnx`
- 입력: 화자 `0000`, `0005`, `0006`의 bucket별 FBank
- 입력 shape: `[1, frames, 80]` float32
- 출력: `[1,192]` embedding
- intra-op threads: **1**
- affinity: CPU 0
- execution: sequential, graph optimization all, CPU arena ON
- memory pattern ON: 같은 session에서 run 1·2·3
- memory pattern OFF: 같은 session에서 run 1·2
- null floor: 같은 입력 shape의 ReduceMean 모델을 bucket마다 fresh process 3회
- latency/RTF 범위: `OrtRun`만 포함; WAV/FBank/model load 제외

`threads=4`를 CPU 하나에 고정하는 조건은 스레드 경합을 만들기 때문에 이 baseline에서
허용하지 않는다. 멀티코어 실험은 별도 CPU set 프로토콜로 수행해야 한다.

## 측정값의 의미

| 출력 지표 | 의미 |
|---|---|
| total/peak RSS | native ORT 프로세스 전체 resident memory |
| PSS/private/shared/anon/file | `/proc/self/smaps_rollup`과 `/proc/self/status` 세부값 |
| model-independent floor | 같은 shape null 모델의 steady-state 메모리 |
| model load resident delta | CAM++ session 직후 − null session 직후 |
| inference resident delta | CAM++ 추론 증가분 − null 추론 증가분 |
| allocator transient upper bound | run 중 새 MaxInUse − run 직전 InUse |
| arena backing growth | `TotalAllocated` 증가량; activation 크기가 아님 |
| offline tensor arena | Custom C runtime의 정확한 정적 tensor arena |
| serialized initializer payload | ONNX initializer 원본 byte 수; resident weight RSS와 같지 않음 |

공개 RSS와 allocator counter만으로 weight, prepack, graph metadata, kernel scratch,
activation tensor를 완전히 분리할 수는 없다. 따라서 결과 JSON의
`exact_ort_activation_tensor_bytes`는 의도적으로 `null`이다. 대신 서로 정의가 다른
resident delta, allocator transient upper bound, offline arena를 함께 기록한다.

## 실행

대상 Linux 디바이스에서:

```bash
bash experiments/baseline_ort_c/build.sh
python3 experiments/baseline_activation_plan/measure_activation_plan.py \
  --save-as-baseline
```

스크립트가 bucket별 null ONNX를 `experiments/baseline_activation_plan/generated/`에
자동 생성한다. `onnx` Python package가 필요하다.

결과:

```text
experiments/baseline_activation_plan/result/<UTC>__native_ort_4bucket_baseline.json
experiments/baseline_activation_plan/result/<UTC>__native_ort_4bucket_baseline.md
experiments/baseline_activation_plan/result/baseline.json   # --save-as-baseline
experiments/baseline_activation_plan/result/baseline.md
```

JSON에는 모든 native raw observation이 남는다. Markdown은 C 최적화 전후 비교에 사용할
요약표다.

## 판정

memory pattern은 다음 조건을 모두 만족할 때 `confirmed`다.

1. ON/OFF의 모든 embedding hash가 동일하다.
2. ON run 2 allocation event가 ON run 1보다 적다.
3. ON run 2 allocation event가 OFF run 2보다 적다.
4. ON run 2와 run 3 event가 동일하다.

run 2에서 arena backing 증가량이 0인 것은 activation이 0이라는 뜻이 아니다. run 1에서
확보한 capacity와 memory pattern을 재사용한다는 뜻이다.
