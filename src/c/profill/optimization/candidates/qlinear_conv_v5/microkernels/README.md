# Microkernels

v5 MAC work lands here in independent steps:

- `qconv_mac_1x1_8x8_real_*`: one call for 8 spatial by 8 output lanes;
- `qconv_mac_3x3_cin32_unroll_*`: shape-specialized 3x3/Cin32 kernel;
- `qconv_mac_3x3_sliding_8x8_*`: interior sliding-window input reuse;
- `qconv_mac_tail_v5_*`: tail paths that preserve v4 semantics.

Until those kernels pass bitwise gates, the v5 dispatch delegates to v4.
