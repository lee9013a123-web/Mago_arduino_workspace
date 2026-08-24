# CAM++ Final-98 speaker embedding model

이 디렉터리의 배포 산출물은 `campp_sv_98.camppmodel`이다. 큰 binary는 생성물이라
Git에 넣지 않고 다음 명령으로 재현한다.

```bash
python3 scripts/5_model/01_build_model_bin.py --force
python3 scripts/5_model/02_verify_model_bin.py \
  --bucket-frames 98 \
  --expect-plan runs/models/campplus/final_v3/weight_residency/98/plan_98.bin \
  --expect-weights runs/models/campplus/final_v3/weight_residency/98/weights_98.bin
```

이 모델은 `[1,98,80]` float32 FBank를 받아 `[1,192]` float32 speaker embedding을
출력한다. WAV decoding, FBank 실행, 긴 음성 window aggregation, enrollment,
cosine score와 threshold 판정은 후속 speaker-verification pipeline의 책임이다.
모델 내부 contract section은 해당 파이프라인이 따라야 할 전처리·후처리 조건을
기록하지만 아직 보정되지 않은 정책이나 threshold를 임의로 확정하지 않는다.
