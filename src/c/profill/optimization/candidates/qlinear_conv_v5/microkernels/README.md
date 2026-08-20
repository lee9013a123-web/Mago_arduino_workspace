# Microkernels

v5 MAC work lands here in independent steps:

- `qconv_mac_1x1_8x8_real_*`: one call for 8 spatial by 8 output lanes;
- `qconv_mac_3x3_cin32_unroll_*`: shape-specialized 3x3/Cin32 kernel;
- `qconv_mac_3x3_sliding_8x8_*`: interior sliding-window input reuse.
  The body walks one input row as `base + column * column_step` and reads
  10 columns to feed 8 outputs, so it is address-correct only when the
  kernel column step equals the output column step, i.e. column stride 1
  with dilation 1.  `planning/qconv_v5_validated_raw_plan.c` decides that
  once per output block and passes it in as `sliding_eligible`; the
  microkernel refuses to run without it.  A stride-2 3x3 therefore takes
  the Cin32 raw body or the shared tail instead;
- `qconv_mac_tail_v5_*`: tail paths that preserve v4 semantics.

Unsupported border/tail cases still fall back to the v4 tail path.
