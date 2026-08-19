# QConv v5 tests

`test_qconv_v5_primitives.c` covers the initial v5 helper ABI: zero-point
fast-path predicates, corrected-bias arithmetic, weight-pack views, stage tag
names, and validated raw plan eligibility.

`test_qconv_v5_address.c` covers the independent address provider: 1x1 and
3x3 pointer walks, explicit padding NULLs, row-wise scheduling, no whole-tile
memset, and V4-to-V5 execution-plan conversion.
