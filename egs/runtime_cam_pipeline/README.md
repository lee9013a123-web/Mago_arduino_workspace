# Runtime CAM speaker-verification pipeline

Arduino/QRB2210의 ALSA microphone에서 음성을 녹음하고, 고정-frame FBank를
Final V3 C runtime에 전달해 192차원 speaker embedding과 cosine score를 만든다.

## 처리 흐름

```text
ALSA mic -> 16 kHz mono WAV -> Kaldi FBank + CMVN -> fixed bucket
         -> bucket plan + weights + schedule -> Final V3 C runtime
         -> L2 embedding -> cosine similarity
```

버킷은 섞어 쓰지 않는다. 기본 package 모드에서 `--bucket 298`이면 298
`.camppmodel`이 선택된다. 이 파일 안에는 298 plan과 weights가 들어 있다.
windowed 모드에서는 298 plan, weights, schedule이 한 번에 선택된다. 어느
모드든 C runtime이 실제로 298 plan을 로드했는지 다시 검사한다.

## 준비

Python 환경에는 `numpy`가 필요하고 보드에는 ALSA `arecord`가 있어야 한다.
`torch`와 `torchaudio`가 있으면 기존 검증 파이프라인과 동일한 Kaldi FBank를
우선 사용하고, 없으면 기존 CAM 코드에서 가져온 NumPy fallback을 사용한다.
fixed-bucket fallback은 Kaldi snip-edge frame 수에 맞게 조정되어 있다. 배포 전
최종 정확도 검증은 `torchaudio` frontend로 수행하는 것을 권장한다. 먼저 장치
이름을 확인한다.

```bash
python3 egs/runtime_cam_pipeline/script/list_microphones.py
```

결과에 맞게 `configs/microphones.json`에 microphone version과 ALSA device를
추가한다. 기본 `arduino_default`는 ALSA `default` device를 사용한다.

### 독립 실행 디렉터리 준비

기본 package 모드는 버킷별 `.camppmodel`을 직접 실행한다.

```bash
bash scripts/4_profill/05_build_final_v3.sh
python3 scripts/5_model/11_build_bucket_models.py --force
python3 egs/runtime_cam_pipeline/script/prepare_runtime.py \
  --mode package --force
```

준비가 끝나면 다음 파일이 모두 이 디렉터리 아래에 존재한다.

```text
runtime/campp_runtime
runtime/assets.json
runtime/models/campp_sv_98.camppmodel
runtime/models/campp_sv_298.camppmodel
runtime/models/campp_sv_498.camppmodel
runtime/models/campp_sv_998.camppmodel
```

이후 등록과 추론은 저장소의 `build/`, `runs/`, `models/`를 참조하지 않는다.
`runtime_cam_pipeline` 디렉터리만 보드의 다른 위치로 복사해도 실행할 수 있다.

weight schedule과 약 1 MiB window를 사용하려면 대신 다음을 준비한다.

```bash
python3 scripts/5_model/08_build_weight_streaming.py --force
bash scripts/5_model/09_build_weight_streaming_98.sh
python3 egs/runtime_cam_pipeline/script/prepare_runtime.py \
  --mode windowed --force
```

`.camppmodel-v1`에는 schedule section이 없으므로 package와 windowed는 서로 다른
배포 모드다. package는 모델 직접 실행, windowed는 더 낮은 weight RSS를 위한
plan+weights+schedule 실행이다.

## 화자 등록

다음 명령은 10초 음성을 5번 연속 녹음한다. 각 녹음은 998-frame 모델로
embedding을 만들며, 개별 L2 normalization 후 평균하고 다시 L2 normalize한
`mean_embedding.f32`를 저장한다.

```bash
cd egs/runtime_cam_pipeline
python3 script/enroll_speaker.py \
  --mic-version arduino_default \
  --speaker-folder lee
```

산출물:

```text
voice/recorded/lee/recording_01.wav ... recording_05.wav
voice/embedded/lee/recording_01.f32 ... recording_05.f32
voice/embedded/lee/mean_embedding.f32
voice/embedded/lee/enrollment.json
```

재등록은 기존 결과를 실수로 덮지 않는다. 의도적으로 다시 만들 때만
`--force`를 붙인다.

## 화자 유사도 추론

등록 폴더 이름 또는 직접 embedding 파일을 선택할 수 있다.

```bash
cd egs/runtime_cam_pipeline
python3 script/verify_speaker.py \
  --mic-version arduino_default \
  --speaker-embedding lee \
  --bucket 298
```

버킷별 녹음 길이는 98=1초, 298=3초, 498=5초, 998=10초다. 결과는 cosine
score, C runtime process peak RSS, logical weight bytes, activation bytes, RTF를
터미널에 표시하고 `runs/inference/<timestamp>/report.json`에도 저장한다.

현재 threshold는 calibration되지 않았다. 따라서 `final score`는 유사도이며
동일/상이 화자를 자동 판정하는 임계값으로 사용하면 안 된다.

실제 녹음 없이 경로·버킷 선택만 확인하려면 각 명령에 `--dry-run`을 붙인다.
