# Runtime 설정

Reference Runtime 검증 파이프라인의 실행 정책을 코드와 분리한다.

- `reference.json`: 정확성 기준선. 전체 Tensor를 dump하고 엄격한 허용 오차를 쓴다.
- `qrb2210.json`: QRB2210 실행 프로필. 현재 backend는 검증이 끝난
  `cpu_reference`이며 ARM 최적화 backend가 완성되면 이 값만 교체한다.

주요 필드:

- `buckets`: 실행할 고정 frame bucket
- `seed`: ORT와 C가 공유하는 입력 생성 seed
- `paths.workspace_root`: 실행별 staging 폴더를 둘 위치
- `diagnostics.dump_tensor_ids`: 현재 정확성 파이프라인은 `"all"`만 지원
- `diagnostics.checkpoint_tensor_ids`: 최종 보고서에서 강조할 Tensor
- `tolerances`: FP32 및 embedding 판정 기준
- `build`: 보드 C compiler와 flags

설정 경로는 저장소 root 기준이다. `workspace_root`는 안전을 위해
`models/compiled/reference` 안에 둘 수 없다.
