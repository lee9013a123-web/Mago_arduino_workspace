# QConv v5 tests

`test_qconv_v5_primitives.c` covers the initial v5 helper ABI: zero-point
fast-path predicates, corrected-bias arithmetic, weight-pack views, stage tag
names, and validated raw plan eligibility.

`test_qconv_v5_address.c` covers the independent address provider: 1x1 and
3x3 pointer walks, explicit padding NULLs, row-wise scheduling, no whole-tile
memset, and V4-to-V5 execution-plan conversion.

`test_qconv_v5_bitwise.c` is the numerical gate for the full-tile MAC bodies.
`test_qconv_candidate.c` compares every candidate mode against the packed
reference, but its shapes never build a full 8-wide spatial tile, so the v5
real 1x1, 3x3 sliding and 3x3/Cin32 bodies all fall through to the shared tail
there and are effectively untested.  This suite drives shapes that enter each
body and compares v4 and v5 byte-for-byte against
`campp_aarch64_qlinear_conv_o4i4`:

- 1x1 full tiles at Cin 32/64 and at column stride 1 and 2;
- 3x3 interior tiles at stride 1, which take the sliding-window body;
- 3x3 at stride 2 and at mixed stride (1, 2), where sliding must stay off;
- 3x3 stride 2 below Cin 32, which must reach the shared tail.

Every case runs twice, with an all-zero and a non-zero weight zero point, so
the zero-point fast path and the corrected-bias path are both covered.
