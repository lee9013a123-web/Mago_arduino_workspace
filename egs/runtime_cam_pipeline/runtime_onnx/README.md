# Generated ONNX Runtime assets

`python3 script/prepare_runtime.py --ort --force` copies the canonical ONNX
model and native FBank binary here. Generated model/binary assets are ignored.
The Python environment must provide `numpy` and `onnxruntime`.

The generated contract is:

```text
runtime_onnx/
  assets.json
  campp_fbank
  licenses/kaldi-native-fbank-LICENSE
  models/campplus_int8_static_qop.onnx
```

`assets.json` binds the model and frontend by SHA-256. One dynamic ONNX model
accepts all fixed frame inputs (98, 298, 498, and 998); `--bucket` controls the
feature shape and recording duration rather than selecting a different model.
