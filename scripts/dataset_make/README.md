# dataset_make

벤치마크·실험이 쓰는 입력 payload와 manifest를 만드는 스크립트를 모은다.
모델 변환이나 kernel 계산은 하지 않는다.

## build_runtime_features.py

`data/multi_speaker`(화자 484명, 화자당 wav 3개, 각 3.0–5.0초)에서 실데이터
FBank payload와 두 manifest를 만든다.

```bash
python3 scripts/dataset_make/build_runtime_features.py
```

산출물:

| 경로 | 내용 |
|---|---|
| `data/multi_speaker_concat/<speaker>.wav` | 이어붙인 source WAV |
| `benchmarks/campplus/features/<input_id>__<N>.f32` | `[1,N,80]` LE float32 |
| `benchmarks/campplus/manifests/runtime_inputs.tsv` | dataset manifest |
| `benchmarks/campplus/manifests/runtime_features.json` | feature manifest |

기본값은 화자 15명 × bucket 4개 = payload 60개다. `--speakers`로 개수를,
`--config`로 bucket 정의를 바꾼다.

### 왜 이어붙이는가

25 ms / 10 ms 기준 bucket frame 수는 오디오 길이와 정확히 대응한다
(`floor((N-400)/160)+1`): 98=1초, 298=3초, 498=5초, 998=10초. 원본 클립이 최대
4.992초라 **단일 클립으로는 498과 998을 채울 수 없다.** 화자별 3개를 이어붙이면
중앙값 12.03초가 되어 408/421명이 10초를 넘긴다. zero-padding 없이 전 구간이
실제 음성이다.

### 고정한 전처리 정책

manifest의 `preprocessing`에 그대로 기록된다.

- **이어붙이기**: 화자별 wav 3개를 파일명 오름차순으로 연결
- **정규화**: int16 → float32 `[-1,1]`, DC 평균 제거 → peak 0.95 정규화 → clip.
  이어붙인 전체 파형에 **1회** 적용한 뒤 자른다
- **Crop**: offset 0에서 bucket 길이만큼. 4개 bucket은 서로의 prefix
- **FBank**: `torchaudio.compliance.kaldi.fbank` — 80 bins, 16 kHz,
  frame 25 ms / shift 10 ms, `dither=0.0`, `energy_floor=0.0`,
  `window_type="hamming"`, `use_energy=False`, `snip_edges=True`
- **CMVN**: time축 평균 감산

FBank 파라미터는 배포 파이프라인
(`egs/pipeline_experiment/core.py::extract_fbank_torchaudio`)에서 그대로 가져왔다.
이 스크립트는 정책을 새로 정하지 않고 복제한 뒤 기록만 한다. `dither=0.0`이라
결정적이며, 같은 입력에서 항상 같은 SHA-256이 나온다.

### 주의

`benchmarks/campplus/manifests/inputs.tsv`는 phase 1이 만드는 **정확도용**
manifest다. 이 스크립트는 그 파일을 건드리지 않고 `runtime_inputs.tsv`를 따로
만든다. `configs/benchmark/runtime_qrb2210.json`의 `dataset_manifest`가 후자를
가리킨다.
