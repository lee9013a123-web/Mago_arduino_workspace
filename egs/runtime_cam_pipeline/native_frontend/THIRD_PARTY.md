# Native FBank dependency

`campp_fbank` uses
[`kaldi-native-fbank`](https://github.com/csukuangfj/kaldi-native-fbank),
version `v1.22.3`, commit
`b09e686fe2084732ddd30d1ef80acfc0f13eaf01`.

The dependency is an Apache-2.0, Kaldi-compatible C++ feature extractor. It is
downloaded into the ignored `.deps/` directory by
`script/build_native_fbank.sh`; downloaded source and build products are not
committed. Its CMake build also downloads the pinned KissFFT revision declared
by that release. Preserve the dependency LICENSE when redistributing source or
the resulting binary.

This is not the full Kaldi toolkit. The deployment binary contains only the
fixed 16 kHz, 80-bin FBank path required by the CAM++ model.
