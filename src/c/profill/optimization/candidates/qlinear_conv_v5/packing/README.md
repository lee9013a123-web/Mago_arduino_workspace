# Packing

Weight packing helpers for v5 live here.  The first production-safe target is
an int8 packed layout that removes runtime transpose/zero-point preparation
without doubling weight storage.
