# Native ORT four-bucket RAM/RTF baseline

| frames | audio | first p50 | warm p50 | warm RTF | peak RSS max | model-attributed RSS | inference resident delta | allocator transient upper | offline arena | pattern |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 98 | 1.0s | 196.58 ms | 192.26 ms | 0.192 | 68.00 MB | 40.58 MB | 12.75 MB | 6.90 MB | 2.01 MB | confirmed |
| 298 | 3.0s | 556.04 ms | 538.44 ms | 0.179 | 97.26 MB | 69.79 MB | 42.04 MB | 21.28 MB | 6.10 MB | confirmed |
| 498 | 5.0s | 917.64 ms | 883.82 ms | 0.177 | 125.95 MB | 98.25 MB | 70.75 MB | 34.94 MB | 10.20 MB | confirmed |
| 998 | 10.0s | 1932.68 ms | 1877.52 ms | 0.188 | 197.68 MB | 170.03 MB | 142.54 MB | 69.73 MB | 20.44 MB | confirmed |

`inference resident delta` is a same-shape null-model differential. It includes activation, scratch, and lazy page faults.

`allocator transient upper` includes allocator-managed scratch/output and is reported only when run 1 establishes a new MaxInUse high-water mark.

The authoritative result, including every native observation, is the sibling JSON file.
