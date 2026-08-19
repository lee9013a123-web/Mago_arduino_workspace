# Dispatch

`qconv_v5_dispatch.*` exposes the candidate runner used by
`qconv_candidate.c`.  It owns the fallback boundary.  MAC kernels below this
directory should receive prevalidated, plain parameter blocks and must not
repeat descriptor checks in the hot loop.
