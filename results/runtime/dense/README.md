# Dense Runtime 검증 결과

Dense block의 누적 Concat을 slab과 VIEW alias로 변경한 실행 결과를 저장한다.

기본 결과 파일:

- `dense_slab_validation.json`: 기존 Tensor Arena와 Dense slab의 Tensor별 bit 비교

다음 명령으로 생성한다.

```bash
python3 scripts/3_runtime/10_export_dense_slab_bundle.py
python3 scripts/3_runtime/11_validate_dense_slab.py
```
