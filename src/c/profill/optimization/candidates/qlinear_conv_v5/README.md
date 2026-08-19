# QLinearConv v5 candidate layout

`qlinear_conv_v5/` is the staging area for bitwise-safe MAC-side follow-up
work after v4.  The initial candidate is wired as an independent
`--qconv-candidate v5` mode and deliberately delegates execution to v4 while
the MAC kernels are developed behind separate files.

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

The v5 rule is strict: numerical transforms must compare retained tensors
against the packed runtime reference before they can replace the v4 path.
Production registry entries are not changed by this candidate.

Address는 독립 provider로 유지한다. `planning/qconv_v5_execution_plan.*`만 V4
Tensor/attribute plan을 address geometry로 변환하며, MAC은 장기적으로
`address/qconv_v5_address_contract.h`의 tile POD만 소비한다. 현재 V5 MAC wrapper가
사용하는 V4 TilePlan ABI는 MAC gate가 끝날 때까지 유지한다.
