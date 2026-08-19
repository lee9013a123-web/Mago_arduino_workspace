# QLinearConv v4 candidate layout

`qlinear_conv_v4/`는 현재 `qlinear_conv/`의 `mac_fixed`와 production
requantization을 복사하지 않고 재사용하면서, 다음 QConv 공통 최적화를 독립적으로
구현·측정하는 staging 영역이다. `--qconv-candidate v4`에서만 활성화되며
production registry에는 직접 등록하지 않는다.

```text
qlinear_conv_v4/
├── dispatch/       # shape/layout별 1x1, 3x3 interior, tail, fallback 선택
├── planning/       # hot loop 밖 validation, tile/offset/parameter 계획
├── parameters/     # multiplier·zero-point·bias의 runtime view
├── microkernels/   # 1x1, 3x3 interior, spatial/channel tail MAC
├── requant/        # 8-lane NEON 및 선택적 fixed-point requant
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

## Implementation status

- 완료: 8-lane float requant와 8-byte channel-packed store
- 완료: layout·scale shape·store capacity를 실행/tile plan으로 hoist
- 완료: rank-3 1x1과 rank-4 3x3 interior 직접 pointer 계획
- 완료: full S8×O8은 기존 fixed intrinsics, boundary/tail은 MAC-v2 fallback
- 완료: `v4` candidate mode, C primitive/통합 bitwise test, board benchmark runner
- 보류: 오프라인 fixed-point multiplier/shift를 bundle에 기록하는 format 확장

fixed-point multiplier는 현재 float `nearbyintf` 결과와 bitwise 동등성이 증명되기
전에는 기본 경로로 선택하지 않는다. binary format을 바꿀 때는 Python writer, C
header/loader와 validator를 같은 변경에서 동기화한다.
