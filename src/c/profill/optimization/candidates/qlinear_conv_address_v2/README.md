# QLinearConv address v2 staging area

이 디렉터리는 V4의 MAC multiply와 dispatch 최적화 작업에 영향을 주지 않고
address 계산만 독립적으로 구현하고 측정하기 위한 staging 영역이다. 현재 V4
빌드에는 연결하지 않는다.

```text
qlinear_conv_address_v2/
├── contracts/                     # address producer와 MAC consumer가 공유하는 POD
├── api/                           # 함수 포인터 없는 address provider API
├── planning/                      # geometry 검증과 loop 밖 파생값 계산
├── kernels/
│   ├── one_by_one/                # 1x1 base + stride pointer walk
│   ├── three_by_three/            # 3x3 interior base + delta pointer walk
│   └── generic/                   # border, padding, spatial/channel tail
└── adapters/
    └── qlinear_conv_v4/           # V4 ExecutionPlan과의 유일한 결합 지점
```

## Dependency boundary

```text
V4 dispatch ──> V4 adapter ──> address API ──> contracts
       │
       └─────────────────────> MAC ──────────> contracts
```

- `address`는 MAC, requant, store, V4 dispatch를 include하지 않는다.
- `MAC`은 address provider를 include하지 않고 `CamppQconvAddressTile`만 소비한다.
- V4 타입을 address 타입으로 변환하는 코드는 `adapters/qlinear_conv_v4/`에만 둔다.
- hot loop에는 provider 함수 포인터를 두지 않는다. dispatch가 경로를 한 번 정한
  뒤 1x1, 3x3, generic 함수를 직접 호출한다.
- padding 위치는 `NULL` input point로 표현한다. 기존 input-zero 처리와 bitwise
  동등성을 유지한다.

## Integration order

1. 기존 `qconv_v4_tile_plan_create()`와 포인터 결과를 단독 테스트에서 비교한다.
2. 1x1과 3x3 interior를 각각 구현하고 generic은 기존 경로를 adapter로 유지한다.
3. address plan을 output-channel block 밖으로 hoist하거나 compact offset cache로
   재사용한다.
4. operator replay의 모든 retained tensor가 bitwise identical일 때만 V4 dispatch에
   연결한다.

기존 V4 파일을 직접 이동하거나 복사하지 않는다. 독립 실험이 실패하면 이
디렉터리만 제거해 기존 candidate를 그대로 보존할 수 있어야 한다.
