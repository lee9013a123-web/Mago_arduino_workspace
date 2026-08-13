# CAM++ benchmark runner

`benchmark_onnx.py`는 `feature[1,T,80]`을 받는 static CAM++ ONNX의 전체 모델 성능을 측정한다.

측정 항목:

- 새 프로세스 cold start 10회
- warm-up 20회 후 warm inference 100회
- mean, p50, p95, p99, 표준편차, 변동계수
- mean/p50/p95/p99 RTF
- 현재 RSS와 peak RSS
- IR에 기록된 custom runtime용 planned arena
- 선택적으로 reference embedding 대비 cosine similarity

## 의존성

Arduino에서 ONNX graph 작업에 사용했던 Python 환경을 활성화한다.

```bash
source ~/workspace/venv/bin/activate
python -c "import numpy, onnxruntime; print(onnxruntime.__version__)"
```

실제 venv 위치가 다르면 해당 환경을 사용한다.

## 3초 모델 성능 측정

실제 feature가 아직 없으면 deterministic synthetic feature를 사용한다. 이는 속도 측정용이며 정확도 평가용이 아니다.

```bash
cd ~/workspace/egs/accelerate_CAM

python3 scripts/1_benchmark/benchmark_onnx.py \
  --config configs/benchmark/qrb2210.json \
  --model results/static/campp_static_298.onnx \
  --ir results/graph/ir_3s.json \
  --output runs/baseline/ort_3s_t1.json \
  --enforce-gates
```

스레드 수를 바꿀 때는 CLI 값이 config보다 우선한다.

```bash
python3 scripts/1_benchmark/benchmark_onnx.py \
  --config configs/benchmark/qrb2210.json \
  --model results/static/campp_static_298.onnx \
  --ir results/graph/ir_3s.json \
  --threads 4 \
  --output runs/baseline/ort_3s_t4.json
```

## 실제 feature와 reference 비교

`--input-f32`는 C Runtime과 같은 바이트를 쓰기 위한 raw little-endian
float32 입력이다. Dynamic canonical model에서는 shape 정보가 없으므로
`--frames`도 함께 지정한다.

`--input-npy`는 `[T,80]` 또는 `[1,T,80]` float feature를 받는다. Static model의 frame 수와 정확히 일치해야 한다.

```bash
python3 scripts/1_benchmark/benchmark_onnx.py \
  --config configs/benchmark/qrb2210.json \
  --model results/static/campp_static_298.onnx \
  --ir results/graph/ir_3s.json \
  --input-npy benchmarks/campplus/features/jonah_id_0_298.npy \
  --reference-npy benchmarks/campplus/reference_outputs/jonah_id_0.npy \
  --output runs/baseline/ort_3s_jonah_t1.json \
  --enforce-gates
```

종료 코드 `2`는 JSON 저장은 끝났지만 CV 3% 또는 cosine 0.999 gate를 통과하지 못했다는 뜻이다.

## 해석 주의사항

- Cold는 매회 새 프로세스지만 Linux page cache를 강제로 비우지 않는다.
- `peak_rss_bytes`는 ONNX Runtime을 포함한 프로세스 전체 메모리다.
- `planned_static_arena_bytes`는 IR에서 계산한 custom runtime용 값이며 ORT가 실제로 그 arena를 사용했다는 뜻이 아니다.
- Synthetic feature 결과로 EER이나 실제 음성 정확도를 판단하면 안 된다.
