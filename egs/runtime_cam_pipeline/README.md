# Runtime CAM speaker-verification pipeline

Arduino/QRB2210의 ALSA microphone에서 음성을 녹음하고, 고정-frame FBank를
Final V3 C runtime에 전달해 192차원 speaker embedding과 cosine score를 만든다.

## 처리 흐름

```text
ALSA mic -> 16 kHz mono WAV -> native Kaldi FBank + CMN -> fixed bucket
         -> bucket plan + weights + schedule -> Final V3 C runtime
         -> L2 embedding -> cosine similarity
```

버킷은 섞어 쓰지 않는다. 기본 package 모드에서 `--bucket 298`이면 298
`.camppmodel`이 선택된다. 이 파일 안에는 298 plan과 weights가 들어 있다.
windowed 모드에서는 298 plan, weights, schedule이 한 번에 선택된다. 어느
모드든 C runtime이 실제로 298 plan을 로드했는지 다시 검사한다.

## 준비

Python 환경에는 `numpy`가 필요하고 보드에는 ALSA `arecord`, CMake, C++17
compiler가 있어야 한다. 배포 경로는 `torch`와 `torchaudio`를 import하지 않는다.
FBank는 Apache-2.0
[`kaldi-native-fbank`](https://github.com/csukuangfj/kaldi-native-fbank)
v1.22.3을 고정해 만든 `campp_fbank`가 담당한다. Kaldi 원본과 이 경량 구현은
모두 C++이며, 여기서는 전체 Kaldi toolkit을 내려받지 않는다.

먼저 네이티브 frontend를 내려받아 빌드한다. 최초 실행에만 GitHub 접근이
필요하며 소스는 ignored `.deps/`, 빌드 결과는 ignored `build/`에 남는다.

```bash
cd egs/runtime_cam_pipeline
bash script/build_native_fbank.sh
```

Torch 의존성과 준비 상태는 무거운 package를 실제 import하지 않고 확인한다.

```bash
python3 script/check_dependencies.py
```

그 다음 장치 이름을 확인한다.

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
runtime/campp_fbank
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
score, pipeline/Python/C Runtime peak RSS, logical weight/activation bytes,
RTF를 터미널에 표시하고 `runs/inference/<timestamp>/report.json`에도 저장한다.

RAM 표의 의미는 다음과 같다.

- `pipeline total peak`: 검증 Python 프로세스와 자식 프로세스(`arecord`,
  `campp_fbank`, `campp_runtime`)의 current RSS 합을 10 ms마다 표본화한 최댓값
- `Python/Torch peak`: 검증 Python 프로세스의 `VmHWM`; 배포 경로에는 Torch가
  없으므로 실제로는 Python/NumPy peak
- `C runtime peak`: C Runtime 프로세스 자체의 `VmHWM`
- `logical weight`, `logical activation`: 모델 계약상 buffer byte 수이며 실제
  resident memory 분해값은 아님

프로세스 RSS 합은 공유 page를 프로세스별로 중복 계산할 수 있으므로 시스템 전체
물리 메모리 사용량과 완전히 같은 값은 아니다.

배포 전에 실제 녹음 WAV로 네이티브 결과와 Torchaudio/Kaldi 정답을 비교할 수
있다. 이 검증 명령에서만 Torch/Torchaudio가 필요하며 보드 배포에는 포함하지
않는다.

```bash
python3 script/validate_native_fbank.py \
  --wav voice/recorded/lee/recording_01.wav \
  --bucket 998
```

현재 threshold는 calibration되지 않았다. 따라서 `final score`는 유사도이며
동일/상이 화자를 자동 판정하는 임계값으로 사용하면 안 된다.

실제 녹음 없이 경로·버킷 선택만 확인하려면 각 명령에 `--dry-run`을 붙인다.

## 가장 단순한 웹 터미널

웹 서버는 Python 표준 라이브러리만 사용한다. 다음 명령을 실행하고 같은 LAN의
브라우저에서 `http://<arduino-ip>:8080`에 접속한다.

```bash
cd egs/runtime_cam_pipeline
python3 script/run_web.py
```

화면의 한 줄 입력칸에 기존 CLI를 그대로 넣고 실행하면 stdout과 stderr가 아래
터미널 영역에 실시간 표시된다.

```bash
python3 script/verify_speaker.py --mic-version arduino_default --speaker-embedding lee --bucket 298
```

웹 입력은 실제 shell이 아니다. 보안을 위해 다음 세 스크립트만 허용하며 pipe,
redirect, `&&`, command substitution은 거부한다.

- `script/enroll_speaker.py`
- `script/verify_speaker.py`
- `script/list_microphones.py`

동시에 하나의 녹음/추론 명령만 실행할 수 있다. 웹 access log는 기본적으로
꺼져 있어 추론 중 불필요한 terminal I/O를 만들지 않는다. LAN 밖에 공개하거나
포트 포워딩하지 않는다.
