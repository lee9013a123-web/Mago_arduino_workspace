# CAM++ acceleration on QRB2210 (Arduino UNO Q)

CAM++ 화자 임베딩을 UNO Q의 QRB2210 위에서 가속하기 위한 작업 공간.
목표는 "GPU 옵션을 켜는 것"이 아니라 **정적 실행 엔진**을 만드는 것이다 —
입력 길이를 버킷으로 고정하고, shape를 컴파일 타임에 전부 풀고,
메모리를 미리 배치하고, 연산을 융합한 뒤 CPU NEON / GPU Vulkan에 배치한다.

대상 모델: `models/campplus_int8_static_qop.onnx` (opset 11, INT8 QOperator,
`feature[B,T,80] -> embedding[B,192]`).

## graph/ — Stage 0: 정적 그래프 생성

동적 shape ONNX를 버킷별 정적 IR로 바꾸고, 그 결과를 리포트로 낸다.

```bash
venv/bin/python campp_acceleration/graph/build_graph.py \
    --seconds 1 3 5 10 \
    --ref-seconds 3 \
    --expand xvector/block2/tdnnd1 head \
    --verify
```

| 파일 | 역할 |
|---|---|
| `graph_ir.py` | shape freeze + ONNX/ORT shape inference, static/runtime 노드 분류, MAC 추정 |
| `analyze.py` | 융합 후보(직선 구간) 탐지, Tensor Arena 오프라인 배치, 스코프 롤업 |
| `verify.py` | 모든 중간 텐서를 graph output으로 노출해 ORT로 1회 실행, 추론 shape와 실측 대조 |
| `render.py` | 블록 단위 / 연산 단위 그래프를 Graphviz DOT + Mermaid로 출력 |
| `build_graph.py` | CLI. 위를 묶어 `out/graph/`에 IR JSON·리포트·다이어그램 생성 |
| `export_static.py` | **실행 가능한** 버킷별 정적 ONNX 생성 (`out/static/`) |

산출물은 `campp_acceleration/out/graph/`:

- `ir_{1,3,5,10}s.json` — 노드·텐서·해결된 shape·arena 오프셋·융합 후보·스코프 통계
- `graph_report.md` — 버킷 표, op 히스토그램, 핫스팟, 융합 후보, arena, shape 검증
- `blocks_*.dot|mmd`, `scope_*.dot|mmd` — 다이어그램 (보드에 graphviz 없음, mmd는 그대로 렌더 가능)

### 실행 가능한 정적 ONNX

`ir_*.json`은 분석용이라 그것만으로는 추론할 수 없다. ORT 안에서 정적화 효과를
실제로 재기 위한 중간 산출물을 따로 만든다 (최종 배포 포맷 아님).

```bash
venv/bin/python campp_acceleration/graph/export_static.py --seconds 1 3 5 10
```

`out/static/campp_static_{98,298,498,998}.onnx` — 각각 다음을 만족한다.

| 조건 | 확인 |
|---|---|
| 입력 shape 완전 고정 | `feature[1, F, 80]`, 다른 프레임 수는 ORT가 `InvalidArgument`로 거부 |
| shape 연산 상수화 | `Shape`/`Gather`/`Constant` 잔존 0개, 1742개 노드가 initializer로 접힘 |
| Dead node 제거 | graph output에서 역방향 도달 불가 노드·initializer 제거 |
| 가중치·양자화 파라미터 유지 | `QLinearConv` 225개, scale 450 / zero_point 448개 그대로 |
| 출력 유지 | `embedding[1, 192]` |

원본 대비 수치 검증: 동일 입력에서 `max|diff| = 0.0`, `cosine = 1.000000000`
(4개 버킷 전부, 비트 단위 동일). 노드 3180 → 1438, 파일 8.52 MB → 8.17 MB.

### 3초 버킷 기준 핵심 수치

- 노드 3180개 중 **1742개(55%)** 가 shape 산술·상수 → 정적 엔진은 실행하지 않음
- 총 1677 M MAC. **head(FCM)가 노드의 3.6%인데 MAC의 42.4%** (노드당 13.69 M),
  xvector는 노드의 96.4%인데 MAC의 57.6% (노드당 0.70 M)
- fan-out 없는 직선 구간 224개 → 융합 시 **커널 호출 1059회 제거** (런타임 노드의 74%)
- Tensor Arena 5.82 MB (텐서마다 개별 할당 시 137.82 MB, 단편화 0%), 가중치 7.08 MB
- shape 검증: 4개 버킷 × 3180 텐서 전부 실측과 일치

### 융합 커널 명칭 (analyze.py의 `FUSION_PATTERNS`가 단일 출처)

이름을 잘못 잡으면 계획에 엉뚱한 커널이 들어가므로, 아래 두 개는 특히 주의한다.

- **`BNAffineReluQuantConvPack`** — BN을 conv 가중치에 접는 것이 **아니다.**
  BN과 conv 사이에 ReLU가 있어 앞으로 접을 수 없고, BN 입력이 dense `Concat`
  출력이라 뒤로도 못 접는다. 융합되는 건 입력 쪽이다: BN affine + ReLU +
  Quantize를 한 패스로 묶어 **int8 conv 입력을 바로 생성**하고, float BN 출력과
  float ReLU 출력을 아예 만들지 않는다.
- **`SegmentContextConsumerFusion`** — 이름을 `SegmentContextView`로 잡으면 틀린다.
  `Expand`만 지우고 결과를 일반 ORT 텐서로 넘기면, 소비자가 요구하는 연속 텐서를
  결국 써야 하므로 아무것도 절약되지 않는다. 소비자(`Add`와 그 뒤 CAM mask conv)까지
  같은 커널에 넣어야 주소 계산만으로 성립한다.

### 알려진 제약

- FP32 원본 CAM++가 로컬에 없다. GPU FP16 Vulkan 경로를 INT8 CPU 경로와
  비교하려면 양자화 이전 모델이 필요하다.
- MAC은 그래프에서 계산한 추정치다. 실측 지연시간 베이스라인은 아직 없다.
