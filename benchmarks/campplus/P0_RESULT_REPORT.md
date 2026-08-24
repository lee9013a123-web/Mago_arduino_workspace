# CAM++ Phase 0 결과 보고서

- 판정일: 2026-08-04
- 검사 대상 Git commit: `a8ee759` (`Add reproducible CAM++ benchmark setup`)
- Benchmark 생성 시 기록된 commit: `d73f39653b58d5ce38c17a116b87afca9a72f17e`
- 대상: Arduino UNO Q의 QRB2210 Linux 영역
- 최종 판정: **미완료 — F0-TARGET, F0-MODEL, F0-EVAL을 아직 승인할 수 없음**

## 1. 생성된 산출물

다음 Phase 0 관리 파일이 생성되어 있다.

- `README.md`
- `benchmark_protocol.md`
- `manifests/inputs.tsv`
- `manifests/trials.tsv`
- `checksums.sha256`
- `onnx_checksums.sha256`
- `selection_summary.json`
- `reference_outputs/README.md`

## 2. 데이터 점검 결과

### 입력 구성

| 구분 | 파일 수 | 화자 구성 | 길이 |
| --- | ---: | --- | --- |
| Enrollment | 12 | jonah, kim, kiuk, ssh 각 3개 | 7.000초 |
| Positive | 20 | jonah, kim, kiuk, ssh 각 5개 | 3.000초 |
| Negative | 45 | 미등록 화자 3명 각 15개 | 59.904초 9개, 60.032초 36개 |
| 합계 | 77 | 등록 4명, 미등록 3명 | 혼합 |

77개 파일은 모두 다음 포맷으로 기록되어 있다.

- sample rate: 16,000 Hz
- channel: mono
- sample width: 16-bit PCM (`2 bytes`)

Manifest의 77개 SHA-256 값은 형식상 정상이고 `checksums.sha256`과 모두 일치한다. 다만 실제 WAV는 PC 저장소에 없으므로 원본 파일의 현재 내용과 일치하는지는 Arduino에서 `sha256sum -c`로 확인해야 한다.

### Trial 구성

| Trial 종류 | 정답 | 개수 |
| --- | ---: | ---: |
| 동일 등록 화자 | 1 | 60 |
| 다른 등록 화자 | 0 | 180 |
| 미등록 화자 | 0 | 540 |
| 합계 |  | 780 |

검사 결과:

- 중복 `input_id`: 0개
- 중복 `trial_id`: 0개
- 존재하지 않는 입력을 가리키는 trial: 0개
- target/trial type 불일치: 0개

따라서 현재 `trials.tsv`의 조합과 라벨 구조는 정상이다.

## 3. ONNX 점검 결과

| 입력 bucket | 모델 | SHA-256 검사 | Graph 정보 |
| --- | --- | --- | --- |
| 1초 / 98 frames | `campp_static_98.onnx` | 통과 | opset 11, `feature[1,98,80]` |
| 3초 / 298 frames | `campp_static_298.onnx` | 통과 | opset 11, `feature[1,298,80]` |
| 5초 / 498 frames | `campp_static_498.onnx` | 통과 | opset 11, `feature[1,498,80]` |
| 10초 / 998 frames | `campp_static_998.onnx` | 통과 | opset 11, `feature[1,998,80]` |

`onnx_checksums.sha256`에 기록된 네 모델의 hash는 PC에 내려받은 실제 ONNX 파일과 모두 일치한다.

기존 graph report에는 정적 ONNX가 원본과 `max|diff|=0.0`, cosine `1.0`이라고 기록되어 있다. 그러나 원본 `models/campplus_int8_static_qop.onnx`, 학습 checkpoint, 해당 checkpoint의 checksum은 현재 저장소에 없으므로 모델 계보 전체가 고정된 상태는 아니다.

## 4. Latency 입력 점검

`latency_selected=1`은 총 15개다.

- 7초 enrollment: 4개
- 3초 positive: 8개
- 약 60초 negative: 3개

이는 정적 ONNX의 1·3·5·10초 bucket과 직접 일치하지 않는다. 특히 약 60초 negative를 그대로 latency 입력으로 사용하는 것은 현재 static model에 넣을 수 없다.

따라서 다음 항목을 추가로 고정해야 한다.

- 원본 WAV에서 1·3·5·10초를 만드는 crop/padding 규칙
- crop 시작 offset
- 짧은 음성의 zero-padding 위치와 값
- 각 길이 bucket에 실제로 사용할 입력 ID
- latency와 정확도 trial을 분리하는 규칙

원본 음성을 복사하지 않고 `source_path`, `start_sample`, `num_samples`, `padding_samples`를 manifest에 기록하는 방식이 적합하다.

## 5. Freeze Gate 판정

| Freeze | 상태 | 근거 |
| --- | --- | --- |
| `F0-TARGET` | **부분 완료** | QRB2210 Linux라는 대상은 기록됐지만 OS, kernel, CPU governor, affinity, 전원, 온도 조건의 실제 snapshot이 없음 |
| `F0-MODEL` | **부분 완료** | 네 static ONNX의 hash와 opset은 확인됐지만 학습 checkpoint, 원본 ONNX hash, export provenance가 없음 |
| `F0-EVAL` | **부분 완료** | 77 inputs와 780 trials는 정상이나 1·3·5·10초 입력 규칙, 무음·잡음·짧은 입력, reference 결과가 없음 |
| Cold/Warm protocol | **절차만 완료** | 20회 warm-up, 100회 반복 등의 규칙은 있으나 실제 측정 결과가 없음 |
| 결과 schema | **미완료** | 결과 JSON 필드와 schema 파일이 없음 |
| Reference outputs | **미완료** | `README.md`만 있고 embedding, score, EER, MinDCF가 없음 |

## 6. Phase 0 종료를 위해 남은 작업

1. Arduino에서 77개 원본 WAV checksum을 실제 검증한다.
2. QRB2210 환경 snapshot을 저장한다.
3. CAM++ checkpoint와 원본 ONNX의 경로·hash·생성 commit을 기록한다.
4. 1·3·5·10초용 crop/padding manifest를 생성한다.
5. 무음·잡음·짧은 음성 입력을 별도 robustness set으로 추가한다.
6. 고정된 PyTorch 또는 기존 기준 runtime으로 FP32/reference embedding을 생성한다.
7. 780 trials의 cosine score, EER, MinDCF를 저장한다.
8. Cold/Warm benchmark 결과 JSON schema를 확정한다.
9. 같은 조건에서 반복 측정하여 허용 편차를 정한다.

## 7. 현재 결론

이번 커밋으로 **평가 데이터의 목록화, trial 설계, 파일 식별, ONNX 식별**까지는 완료됐다. 그러나 정확도 기준값과 장치 조건, 실제 길이별 입력, 결과 schema가 없으므로 이 상태에서 최적화 전후 성능을 공정하게 확정 비교할 수는 없다.

따라서 Phase 1로 바로 넘어가기보다 위 미완료 항목을 채운 뒤 `F0-TARGET`, `F0-MODEL`, `F0-EVAL`을 승인해야 한다.
