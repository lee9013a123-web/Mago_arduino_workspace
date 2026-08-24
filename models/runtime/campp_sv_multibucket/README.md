# CAM++ fixed-bucket model set

이 디렉터리는 하나의 모호한 multibucket binary 대신, 기존 `camppmodel-v1`
loader가 독립적으로 검증할 수 있는 네 모델과 선택 manifest를 사용한다.

```text
campp_sv_multibucket.json
campp_sv_98.camppmodel
campp_sv_298.camppmodel
campp_sv_498.camppmodel
campp_sv_998.camppmodel
```

모델 binary는 생성물이므로 Git에 넣지 않는다. bucket별 static plan과 weights를
준비한 뒤 다음 명령으로 재생성한다.

```bash
python3 scripts/5_model/11_build_bucket_models.py --force
```

FBank payload는 `[1,T,80]` little-endian float32여야 한다. 실행기는 파일 크기로
`T`를 계산하고 manifest, SHA-256, package 입력 계약과 Final V3 runtime suite를
검사한 뒤 정확히 일치하는 모델만 실행한다.

```bash
python3 scripts/5_model/12_run_bucket_model.py \
  --input benchmarks/campplus/features/multi__speaker_0000__298.f32 \
  --embedding-output runs/model_packages/example_298_embedding.f32 \
  --preflight-only

python3 scripts/5_model/12_run_bucket_model.py \
  --input benchmarks/campplus/features/multi__speaker_0000__298.f32 \
  --embedding-output runs/model_packages/example_298_embedding.f32
```

QRB2210에서 네 package가 분리 plan/weights와 모든 retained tensor까지 bitwise
동일한지 한 번에 검증한다.

```bash
bash scripts/4_profill/05_build_final_v3.sh
python3 scripts/5_model/13_validate_bucket_models.py --preflight-only
python3 scripts/5_model/13_validate_bucket_models.py --force
```

이 방식은 package weight를 full-resident로 로드한다. windowed mmap RSS 실험은
`plan + weights + weight_schedule` sidecar 경로를 계속 사용한다.
