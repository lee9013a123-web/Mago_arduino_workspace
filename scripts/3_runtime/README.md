# Phase 3: Reference Runtime

이 폴더에는 `src/python/runtime_bundle_exporter/`와 `src/c/runtime/`를 순서대로 실행하는 얇은 CLI와 shell script만 둔다. 모델 변환 로직이나 C kernel 구현을 이 폴더에 직접 작성하지 않는다.

## 예정 실행 순서

1. `01_export_reference_bundle.py`
   - 정적 ONNX와 graph IR을 읽는다.
   - `weights.bin`, bucket별 `plan_*.bin`, `manifest.json`을 생성한다.
2. `02_dump_ort_references.py`
   - 동일 입력에 대한 ORT 중간 Tensor와 최종 embedding을 저장한다.
3. `03_build_reference_runtime.sh`
   - Arduino에서 CMake로 Reference Runtime을 빌드한다.
4. `04_run_reference_runtime.sh`
   - 1, 3, 5, 10초 bucket을 C Runtime으로 실행한다.
5. `05_compare_runtime_outputs.py`
   - ORT와 C Runtime 출력을 비교하고 첫 불일치 operator를 찾는다.

스크립트가 생성하는 원시 출력은 `runs/runtime/`, 통과 결과와 보고서는 `results/runtime/`에 저장한다.
