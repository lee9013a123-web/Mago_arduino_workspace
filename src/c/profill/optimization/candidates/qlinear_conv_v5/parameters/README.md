# Parameters

This directory contains helpers for bitwise-safe algebraic fast paths:

- all-zero weight zero-points;
- precomputed weight sums;
- corrected bias views for `sum(x * w) + corrected_bias`.

The initial helpers do not change runtime behavior.  They make the invariants
testable before a microkernel consumes them.
