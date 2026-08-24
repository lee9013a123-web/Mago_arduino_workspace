# Runtime CAM speaker-verification pipeline

Arduino/QRB2210의 ALSA microphone에서 음성을 녹음하고, 고정-frame FBank를
Final V3 C runtime에 전달해 192차원 speaker embedding과 cosine score를 만든다.

## 처리 흐름

```text
ALSA mic -> 16 kHz mono WAV -> Kaldi FBank + CMVN -> fixed bucket
         -> bucket plan + weights + schedule -> Final V3 C runtime
         -> L2 embedding -> cosine similarity
```

버킷은 섞어 쓰지 않는다. `--bucket 298`이면 manifest의 298 plan, 298 weights,
298 schedule이 한 번에 선택되며 C runtime이 실제로 298 plan을 로드했는지 다시
검사한다.

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

버킷별 plan/weights/schedule과 weight-streaming runtime을 준비한다.

```bash
python3 scripts/5_model/08_build_weight_streaming.py --force
bash scripts/5_model/09_build_weight_streaming_98.sh
```

## 화자 등록

다음 명령은 10초 음성을 5번 연속 녹음한다. 각 녹음은 998-frame 모델로
embedding을 만들며, 개별 L2 normalization 후 평균하고 다시 L2 normalize한
`mean_embedding.f32`를 저장한다.

```bash
python3 egs/runtime_cam_pipeline/script/enroll_speaker.py \
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
python3 egs/runtime_cam_pipeline/script/verify_speaker.py \
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
