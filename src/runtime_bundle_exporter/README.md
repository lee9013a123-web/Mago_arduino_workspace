# Runtime bundle exporter

정적 CAM++ ONNX와 bucket별 graph IR을 보드용 Reference Runtime 입력으로 변환하는 Python 구현 영역이다.

## 입력

- `models/source/campplus_int8_static_qop.onnx`
- `results/static/campp_static_{98,298,498,998}.onnx`
- `results/graph/ir_{1,3,5,10}s.json`

## 출력

- `models/compiled/reference/weights.bin`
- `models/compiled/reference/execution_plans/plan_{98,298,498,998}.bin`
- `models/compiled/reference/manifest.json`

## 예정 모듈

- `binary_format_schema.py`: Python과 C가 공유할 binary 필드 규격
- `static_model_reader.py`: ONNX node, initializer, attribute 추출
- `graph_ir_reader.py`: graph IR의 Tensor shape와 실행 순서 로딩
- `tensor_table_builder.py`: Tensor ID와 descriptor 생성
- `operator_table_builder.py`: runtime operator table 생성
- `weight_blob_writer.py`: weights binary 직렬화
- `execution_plan_writer.py`: bucket별 execution plan 직렬화
- `bundle_manifest_writer.py`: 모델 hash와 산출물 checksum 기록

CLI 인자 처리와 실행 순서는 `scripts/3_runtime/`에 두며, 재사용 가능한 변환 로직만 이 폴더에 둔다.
