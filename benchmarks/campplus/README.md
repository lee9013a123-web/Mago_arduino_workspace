# CAM++ Benchmark Dataset

이 폴더는 음성 파일을 복사하지 않고 `/home/arduino/workspace/data` 아래 기존 데이터의 선택 규칙과 검증 정보를 저장한다.

## 선택한 데이터

| Split | 파일 수 | 용도 |
| --- | ---: | --- |
| Enrollment | 12 | 등록 embedding 생성 |
| Positive | 20 | 등록 화자의 positive test |
| Negative | 45 | 미등록 화자의 impostor test |

등록 화자: jonah, kim, kiuk, ssh

## 제외한 데이터

- `hey_jarvis.wav`: Wakeword 입력
- `speaker_enroll_wakeword/`: Wakeword 실험용
- `kss_cer_eval_bundle/`: ASR CER 평가용
- `kss_cer_eval_bundle.tar.gz`: 데이터 archive

## 파일

- `manifests/inputs.tsv`: 선택한 음성과 metadata
- `manifests/trials.tsv`: EER용 enrollment/test 쌍과 정답
- `checksums.sha256`: 선택한 음성 파일 checksum
- `onnx_checksums.sha256`: 현재 static ONNX checksum
- `benchmark_protocol.md`: 고정 측정 절차
- `selection_summary.json`: 생성 결과 요약
- `reference_outputs/`: PyTorch 기준 embedding 저장 위치

## Checksum 검증

데이터 루트에서 실행한다.

```bash
cd /home/arduino/workspace/data
sha256sum -c /home/arduino/workspace/egs/accelerate_CAM/benchmarks/campplus/checksums.sha256
```

## Latency subset

- `enroll__speaker_enroll_7sec_jonah_en_0` — `speaker_enroll_7sec/jonah_en_0.wav`
- `enroll__speaker_enroll_7sec_kim_en_0` — `speaker_enroll_7sec/kim_en_0.wav`
- `enroll__speaker_enroll_7sec_kiuk_en_0` — `speaker_enroll_7sec/kiuk_en_0.wav`
- `enroll__speaker_enroll_7sec_ssh_en_0` — `speaker_enroll_7sec/ssh_en_0.wav`
- `negative__speaker_negative_unregistered_speaker_0000_speaker_0000_00000` — `speaker_negative_unregistered/Speaker_0000/Speaker_0000_00000.wav`
- `negative__speaker_negative_unregistered_speaker_0001_speaker_0001_00000` — `speaker_negative_unregistered/Speaker_0001/Speaker_0001_00000.wav`
- `negative__speaker_negative_unregistered_speaker_0002_speaker_0002_00000` — `speaker_negative_unregistered/Speaker_0002/Speaker_0002_00000.wav`
- `positive__speaker_identify_3sec_positive_jonah_id_0` — `speaker_identify_3sec_positive/jonah_id_0.wav`
- `positive__speaker_identify_3sec_positive_jonah_id_2` — `speaker_identify_3sec_positive/jonah_id_2.wav`
- `positive__speaker_identify_3sec_positive_kim_id_0` — `speaker_identify_3sec_positive/kim_id_0.wav`
- `positive__speaker_identify_3sec_positive_kim_id_2` — `speaker_identify_3sec_positive/kim_id_2.wav`
- `positive__speaker_identify_3sec_positive_kiuk_id_0` — `speaker_identify_3sec_positive/kiuk_id_0.wav`
- `positive__speaker_identify_3sec_positive_kiuk_id_2` — `speaker_identify_3sec_positive/kiuk_id_2.wav`
- `positive__speaker_identify_3sec_positive_ssh_id_0` — `speaker_identify_3sec_positive/ssh_id_0.wav`
- `positive__speaker_identify_3sec_positive_ssh_id_2` — `speaker_identify_3sec_positive/ssh_id_2.wav`
