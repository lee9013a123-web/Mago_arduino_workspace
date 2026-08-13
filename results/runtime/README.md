# Runtime 검증 결과

Runtime 검증 결과를 목적별 하위 폴더에 보존한다.

- `c_runtime_compare/`: 기존 ORT 대 C Runtime 비교, End-to-end 및 Tensor Arena 검증
- `dense/`: Dense concat 제거와 slab/view alias 최적화 검증

최종 파일:

- `c_runtime_compare/operator_validation.json`: 구현 opcode와 bucket별 수치 판정
- `c_runtime_compare/end_to_end_validation.json`: 네 bucket의 End-to-end 판정
- `c_runtime_compare/reference_runtime_report.md`: 기존 기준선 고정 보고서
- `dense/dense_slab_validation.json`: 기존 Arena와 Dense slab 실행 비교

세 파일은 다음 명령으로 다시 생성한다.

```bash
python3 scripts/3_runtime/07_freeze_reference_results.py
python3 scripts/3_runtime/10_export_dense_slab_bundle.py
python3 scripts/3_runtime/11_validate_dense_slab.py
```

`baseline_frozen`은 bundle과 네 bucket의 전체 Operator 증거가 고정됐다는 뜻이다.
`phase4_ready`는 그와 별도로 모든 수치 기준과 End-to-end 증거까지 통과했는지를
뜻한다. 알려진 실패를 숨기기 위해 두 판정을 하나로 합치지 않는다.
