# Runtime 검증 결과

Reference Runtime 검증 증거와 Phase 4 정확도 기준선을 장기 보존한다.

최종 파일:

- `operator_validation.json`: 구현 opcode, bucket·opcode별 수치 판정과 입력 증거 SHA-256
- `end_to_end_validation.json`: 네 bucket의 실행, block 경계, embedding 및 Phase 4 판정
- `reference_runtime_report.md`: Git/model/format hash, 허용 오차, 제한 사항을 요약한 보고서

세 파일은 다음 명령으로 다시 생성한다.

```bash
python3 scripts/3_runtime/07_freeze_reference_results.py
```

`baseline_frozen`은 bundle과 네 bucket의 전체 Operator 증거가 고정됐다는 뜻이다.
`phase4_ready`는 그와 별도로 모든 수치 기준과 End-to-end 증거까지 통과했는지를
뜻한다. 알려진 실패를 숨기기 위해 두 판정을 하나로 합치지 않는다.
