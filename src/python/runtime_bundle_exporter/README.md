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

## 모듈 구조

- `format/binary_format_schema.py`: Python과 C가 공유할 binary 필드 규격
- `reader/static_model_reader.py`: ONNX node, initializer, attribute 추출
- `reader/graph_ir_reader.py`: graph IR의 Tensor shape와 실행 순서 로딩
- `builder/tensor_table_builder.py`: Tensor ID와 descriptor 생성
- `builder/operator_table_builder.py`: runtime operator table 생성
- `planner/tensor_arena_planner.py`: bucket별 activation lifetime과 Arena offset 계산
- `planner/dense_slab_planner.py`: Dense Concat chain을 slab-backed VIEW로 변환
- `planner/cache_layout_planner.py`: NTC/NHWC channel padding, stride와 layout VIEW 계획
- `planner/weight_packing_planner.py`: QLinearConv O4I4 오프라인 packing과 복원 검증
- `planner/fusion_planner.py`: 독립 fusion flag, dead Tensor 제거, Tensor ID와
  lifetime 재생성, operator별 kernel ID와 memory traffic 계획
- `writer/weight_blob_writer.py`: weights binary 직렬화
- `writer/execution_plan_writer.py`: bucket별 execution plan 직렬화
- `writer/bundle_manifest_writer.py`: 모델 hash와 산출물 checksum 기록
- `runtime_ir.py`: ONNX와 독립적인 Runtime 자료구조

CLI 인자 처리와 실행 순서는 `scripts/3_runtime/`에 두며, 재사용 가능한 변환 로직만 이 폴더에 둔다.
