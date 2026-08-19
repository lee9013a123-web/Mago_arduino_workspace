# QLinearConv v4 candidate layout

`qlinear_conv_v4/`는 현재 `qlinear_conv/`의 `mac_fixed`와 production
requantization을 복사하지 않고 재사용하면서, 다음 QConv 공통 최적화를 독립적으로
구현·측정하기 위한 staging 영역이다. production registry에는 직접 등록하지
않는다.

```text
qlinear_conv_v4/
├── dispatch/       # shape/layout별 1x1, 3x3 interior, tail, fallback 선택
├── planning/       # hot loop 밖 validation, tile/offset/parameter 계획
├── parameters/     # multiplier·zero-point·bias의 runtime view
├── microkernels/   # 1x1, 3x3 interior, spatial/channel tail MAC
├── requant/        # 8/16-lane NEON 및 선택적 fixed-point requant
└── store/          # channel-packed 연속 store와 alias-safe tail
```

## Dependency direction

```text
dispatch
  ├── planning ── parameters
  ├── microkernels
  └── requant ── store

unsupported case ──> ../qlinear_conv/mac_fixed 또는 production fallback
```

`planning`만 model/operator/Tensor descriptor를 해석한다. microkernel,
requant, store는 검증이 끝난 작은 POD parameter만 받아 hot loop에서 attribute,
shape, layout 검사를 반복하지 않는다.

## Planned implementation order

1. `requant/`의 8-lane 변환과 `store/`의 8-byte 연속 store
2. `planning/`에서 layout·scale·tail 검사를 loop 밖으로 hoist
3. `microkernels/`의 1x1 전용 경로
4. 3x3 interior와 boundary/tail 전용 경로
5. `parameters/`와 Python exporter를 연결한 오프라인 multiplier 계획
6. stage microbench, retained-tensor bitwise, Quick E2E 순으로 gate

fixed-point multiplier는 현재 float `nearbyintf` 결과와 bitwise 동등성이 증명되기
전에는 기본 경로로 선택하지 않는다. binary format을 바꿀 때는 Python writer, C
header/loader와 validator를 같은 변경에서 동기화한다.

