# CAM++ Benchmark Protocol

## 1. 고정 대상

- Device: Arduino UNO Q, QRB2210 Linux 영역
- Repository commit: `d73f39653b58d5ce38c17a116b87afca9a72f17e`
- Data root: `/home/arduino/workspace/data`
- Registered speakers: jonah, kim, kiuk, ssh
- Sample rates found: 16000 Hz
- Channel counts found: 1
- Input manifest: `manifests/inputs.tsv`
- Trial manifest: `manifests/trials.tsv`
- Input checksums: `checksums.sha256`
- ONNX checksums: `onnx_checksums.sha256`

모델 checkpoint, export script commit, ONNX opset은 ONNX 검증 단계에서 추가로 기록한다.

## 2. 데이터 선택 규칙

### 정확도·EER

- `speaker_enroll_7sec/`의 모든 WAV를 enrollment로 사용한다.
- `speaker_identify_3sec_positive/`의 모든 WAV를 positive test로 사용한다.
- `speaker_negative_unregistered/`의 모든 WAV를 unregistered negative로 사용한다.
- Enrollment와 같은 등록 화자의 positive pair는 `target=1`이다.
- 다른 등록 화자의 positive pair는 `registered_impostor`, `target=0`이다.
- 미등록 화자 pair는 `unregistered_impostor`, `target=0`이다.

### Latency

- 화자별 첫 번째 7초 enrollment 파일을 선택한다.
- 등록 화자별 `id_0`, `id_2` positive 파일을 선택한다.
- 미등록 그룹별 첫 번째 파일을 선택한다.
- 정확한 목록은 `inputs.tsv`의 `latency_selected=1` 행이다.

### 제외

- Wakeword와 ASR CER 데이터는 CAM++ 기본 benchmark에서 제외한다.
- 원본 음성은 수정·복사하지 않는다.

## 3. 정확도 측정

1. 모든 enrollment와 test 입력에서 embedding을 생성한다.
2. Embedding에 모델의 기존 normalization 규칙을 동일하게 적용한다.
3. `trials.tsv`의 각 pair에 cosine similarity를 계산한다.
4. Target·non-target score로 EER과 MinDCF를 계산한다.
5. PyTorch FP32를 reference로 저장한다.
6. ONNX FP32, custom FP32, INT8 결과를 같은 trial list로 비교한다.

## 4. 지연시간 측정

### Cold start

- 새 프로세스에서 모델 load부터 첫 embedding 출력까지 측정한다.
- 최소 10회 별도 프로세스로 반복한다.

### Warm inference

- 같은 프로세스에서 20회 warm-up한다.
- 각 latency 입력을 100회 측정한다.
- p50, p95, p99, 평균, 표준편차, RTF를 기록한다.
- 모델 load와 audio file read 시간은 inference 시간과 분리한다.

### Thread

- CPU 1·2·4 thread를 각각 측정한다.
- 동일 실행에서 thread 수를 섞지 않는다.
- CPU affinity, governor, 보드 전원 조건을 결과에 기록한다.

## 5. 입력 길이 실험

- 정확도 평가는 원본 native duration을 사용한다.
- Static shape 성능 실험은 98·298·498·998 frame 모델의 실제 시간 대응 관계를 export script에서 먼저 확인한다.
- 길이 실험용 crop·zero padding은 정확도 trial과 분리한다.
- crop 시작점과 padding 정책은 config에 고정한다.

## 6. 기록 지표

- Cold start latency
- Warm p50·p95·p99 latency
- RTF
- Peak RSS와 arena size
- CPU thread 수와 cache miss
- GPU kernel·동기화 시간(사용 시)
- Embedding cosine similarity
- EER·MinDCF
- 온도와 throttling 여부

## 7. 결과 유효 조건

- `sha256sum -c checksums.sha256`가 모두 통과한다.
- Model checksum과 Git commit이 결과에 기록되어 있다.
- Warm-up·반복 수·thread 수가 기록되어 있다.
- Reference embedding 생성 실패가 없다.
- 동일 조건 반복 측정의 변동 원인을 설명할 수 있다.
