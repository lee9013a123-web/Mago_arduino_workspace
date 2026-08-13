# Cache layout·packing 검증

`cache_packed_validation.json`은 Dense slab bundle과 cache-packed bundle을 다음
고정 입력에서 비교한 결과다.

- `benchmarks/campplus/features/multi__speaker_0000__98.f32`
- `benchmarks/campplus/features/multi__speaker_0005__98.f32`
- `benchmarks/campplus/features/multi__speaker_0006__98.f32`

각 Operator 출력은 stride를 반영한 논리적 C 순서로 저장해 tensor ID별 bitwise
비교한다. VIEW로 변환되어 실행 출력이 사라진 Transpose·Unsqueeze·Squeeze·
singleton Reshape Tensor 108개만 비교 대상에서 제외한다. 남은 1,278개 출력은
모두 bitwise 비교한다. 대용량 중간 dump는 비교 직후 삭제하며 `--keep-dumps`를
지정한 경우에만 보존한다.

```bash
python3 scripts/3_runtime/12_export_cache_packed_bundle.py
python3 scripts/3_runtime/13_validate_cache_packed.py
```
