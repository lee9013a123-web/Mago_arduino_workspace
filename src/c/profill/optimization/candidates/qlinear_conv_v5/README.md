# QLinearConv v5 candidate layout

`qlinear_conv_v5/` is the staging area for bitwise-safe MAC and address-side
follow-up work after v4.  The candidate is wired as an independent
`--qconv-candidate v5` mode. Unsupported blocks fall back at the dispatch
boundary while each v5 path is admitted behind its own correctness gate.

```text
qlinear_conv_v5/
├── dispatch/          # candidate entry point and fallback boundary
├── planning/          # one-time validation and raw-kernel eligibility
├── address/           # row schedule, pointer walk, padding/border provider
├── parameters/        # zero-point and corrected-bias helpers
├── microkernels/      # real 8x8, 3x3/Cin32, sliding-window MAC work
├── packing/           # packed weight view helpers
└── instrumentation/   # hotspot stage tags for v5-only profiling
```

Implemented MAC-side paths:

- validated raw dispatch without the per-tile `try_tile` guard scan;
- all-zero weight zero-point fast path;
- corrected bias for `sum(x * w) + (bias - input_zero * sum(w))`;
- real 8-spatial by 8-output raw body for 1x1;
- 3x3/Cin32 raw fallback;
- 3x3 sliding-window input reuse for interior tiles, admitted only at
  column stride 1 (see `microkernels/README.md`).

The v5 rule is strict: numerical transforms must compare retained tensors
against the packed runtime reference before production registry entries are
changed.

Address는 독립 provider로 유지한다. `planning/qconv_v5_execution_plan.*`만 V4
Tensor/attribute plan을 address geometry로 변환하며, MAC은 장기적으로
`address/qconv_v5_address_contract.h`의 tile POD만 소비한다. 현재 V5 MAC wrapper는
아직 V4 TilePlan ABI를 사용한다. 따라서 address 코드는 빌드·단독 검증되지만,
dispatch의 address provider 교체 시점은 MAC bitwise gate 이후다.
