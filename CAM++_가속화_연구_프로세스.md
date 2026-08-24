# CAM++ 가속화 연구 프로세스

> Arduino UNO Q의 QRB2210에서 CAM++를 PyTorch 없이 실행하고, 정적 offset·NEON·INT8·CPU/GPU 분할을 통해 지연시간과 메모리 이동을 최소화하기 위한 연구 문서

---

## 문서 정보

| 항목 | 내용 |
| --- | --- |
| 주 개발 기기 | Arduino UNO Q — QRB2210 Linux 영역 |
| 1차 대상 모델 | CAM++ speaker embedding model |
| 2차 검증 기기 | Galaxy Note9 — Exynos 9810 |
| 후속 확장 | Whisper encoder → Zipformer |
| 보조 프로세서 | STM32U585 — Wakeword·VAD·센서 처리 |
| 최종 실행 환경 | Native C/C++ runtime, AArch64 NEON |
| 기준 모델 형식 | ONNX |
| 최종 배포 형식 | QRB2210 전용 packed binary |
| 문서 기준일 | 2026-08-04 |

---

# 1. 프로젝트 한눈에 보기

## 1.1 최종 목표

PyTorch를 보드에 설치하지 않고 다음 파이프라인을 완성한다.

```text
PyTorch 학습 모델
    ↓ 1회 export
ONNX 공통 원본
    ↓ PC 오프라인 컴파일러
campplus_qrb2210_int8.bin
    ↓ UNO Q
Native C/C++ + NEON inference runtime
```

최종 런타임은 다음 특징을 가져야 한다.

- 추론 중 Tensor별 `malloc/free`가 없다.
- 모든 activation은 하나의 정적 Tensor Arena에서 offset으로 관리한다.
- CAM++ dense connection의 반복 `concat` 복사가 없다.
- 가중치는 QRB2210 NEON 커널이 읽는 순서로 미리 패킹한다.
- CPU·GPU를 사용할 경우 큰 서브그래프 단위로만 전환한다.
- 모델 로딩 후 가중치 재배치나 그래프 최적화를 수행하지 않는다.
- 성능뿐 아니라 speaker embedding 품질과 EER을 함께 검증한다.

## 1.2 핵심 연구 질문

1. 현재 CAM++에서 실제 시간을 가장 많이 사용하는 연산은 무엇인가?
2. 계산량과 메모리 이동량 중 어느 쪽이 주된 병목인가?
3. Dense connection의 `concat`을 offset 기반 slab으로 바꾸면 얼마나 개선되는가?
4. FP32 NEON, INT8 NEON, GPU의 손익분기점은 어디인가?
5. 1·2·4개 CPU thread 중 연산별 최적값은 무엇인가?
6. CPU와 GPU가 공유 메모리를 사용해도 동기화 비용이 성능을 상쇄하는가?
7. 모델별 정적 실행 계획이 ONNX Runtime보다 얼마나 빠르고 작아지는가?
8. 같은 실행 엔진 구조를 Note9와 Whisper encoder에 얼마나 재사용할 수 있는가?

## 1.3 범위

### 포함

- CAM++ inference graph 분석
- PyTorch → ONNX 변환과 수치 검증
- ONNX Runtime 기준선 측정
- 정적 graph compiler
- 정적 offset 기반 Tensor Arena
- AArch64 NEON FP32·INT8 커널
- 가중치 사전 패킹
- 연산 융합
- OpenCL/Vulkan GPU 실험
- CPU/GPU 자동 선택
- QRB2210 전용 binary 생성
- Note9 호환성 검증

### 현재 제외

- CAM++ 학습 자체의 재설계
- FPGA·Verilog 가속기 구현
- QRB2210의 물리 RAM 주소 직접 제어
- `/dev/mem`을 사용한 물리 주소 접근
- 검증되지 않은 LPASS DSP/QNN 의존
- STM32U585에서 CAM++ 전체 실행

---

# 2. 하드웨어 이해

## 2.1 QRB2210

| 구성 | 확인된 사양 | 연구상 의미 |
| --- | --- | --- |
| CPU | 64-bit A53 계열 4코어, 최대 2.0GHz | AArch64 NEON 사용, thread 수 튜닝 필요 |
| CPU L2 | 공유 512KB | 전체 모델이 아닌 작은 tile 단위 처리 필요 |
| GPU | Adreno 702, 845MHz | 큰 Conv·GEMM 후보, 작은 연산은 호출 비용 주의 |
| 메모리 | LPDDR4X, 2×16-bit, 1804MHz, 최대 4GB | CPU·GPU가 같은 DRAM 대역폭을 공유 |
| 메모리 구조 | Unified memory | 별도 VRAM 복사는 없지만 cache flush·barrier는 필요 |
| OpenCL | 2.0 | 초기 GPU compute 실험에 사용 가능 |
| Vulkan | 하드웨어 1.1, 현재 UNO Q 드라이버는 1.0 계열 | 공통 GPU backend는 보수적인 Vulkan 기능만 사용 |

LPDDR4X 사양으로 계산한 이론 대역폭은 약 14.4GB/s이지만, 실제 값은 CPU·GPU·디스플레이의 경쟁과 메모리 접근 패턴에 따라 더 낮다. 공식 사양만으로 성능을 추정하지 않고 반드시 보드에서 측정한다.

## 2.2 STM32U585의 역할

STM32U585는 QRB2210과 역할을 분리한다.

| QRB2210 | STM32U585 |
| --- | --- |
| CAM++ | Wakeword |
| Whisper encoder | VAD |
| Zipformer | 센서 전처리 |
| Linux·CPU·GPU 실행 | 항상 켜진 저전력 실시간 처리 |

STM32U585에서는 linker script를 사용해 SRAM section을 직접 지정할 수 있지만, CAM++ 전체 모델의 실행 대상은 아니다.

## 2.3 FPGA와의 차이

UNO Q에는 FPGA 논리 블록이 없으므로 Verilog로 새로운 MAC 배열이나 가속기를 만들 수 없다. 이번 연구의 목표는 기존 CPU·GPU가 NPU처럼 효율적으로 동작하도록 다음 소프트웨어 요소를 구현하는 것이다.

- 정적 메모리 계획
- 가중치 사전 배치
- 캐시 기반 tiling
- SIMD·NEON 커널
- 연산 융합
- 정적 스케줄링
- CPU/GPU coarse-grained dispatch

---

# 3. 성공 기준

## 3.1 반드시 기록할 지표

| 분류 | 지표 |
| --- | --- |
| 지연시간 | Cold start, warm p50, p95, p99 |
| 처리량 | RTF, 초당 처리 음성 길이 |
| 메모리 | Peak RSS, arena 크기, 모델 상주 크기 |
| CPU | thread 수, cycles, instructions, cache miss |
| GPU | kernel 시간, dispatch 횟수, CPU/GPU 전환 횟수 |
| 메모리 이동 | 연산별 read/write 추정량, backend 경계 전달 바이트 |
| 정확도 | Embedding cosine similarity, EER, MinDCF |
| 안정성 | 30분 이상 반복 실행, 메모리 증가 여부 |
| 열·전력 | 온도, clock throttling, 장시간 지연시간 변화 |

## 3.2 초기 합격 기준

아래 값은 baseline 측정 후 조정할 수 있는 초기 기준이다.

- FP32 custom runtime 출력과 ONNX 기준 출력의 cosine similarity ≥ 0.999
- INT8 적용 후 EER 악화 ≤ 0.1%p를 우선 목표로 설정
- Warm benchmark의 반복 변동계수 ≤ 3%
- 추론 반복에 따른 RSS 증가 없음
- 추론 loop 내부 동적 메모리 할당 0회
- CPU/GPU 경계 전환은 전체 graph에서 최대 2회
- 최적화는 단독 변경 전후를 비교할 수 있어야 함

## 3.3 이 연구에서 "고정"의 의미

이 연구는 처음부터 모든 값을 추측해서 고정하지 않는다. 각 단계에서 후보를 측정한 뒤, 정확도와 성능 기준을 통과한 항목을 하나씩 **Freeze**한다.

```text
후보 생성
→ 정확도 검증
→ 성능 비교
→ 최적안 선택
→ Freeze ID 발급
→ 다음 단계에서는 고정값으로 사용
```

고정된 항목은 최종적으로 QRB2210 전용 binary에 들어가며, 보드의 inference loop에서는 해당 항목을 다시 분석하거나 선택하지 않는다.

고정에는 세 종류가 있다.

| 종류 | 의미 | 예시 |
| --- | --- | --- |
| 연구 기준 고정 | 모든 실험이 공유해야 하는 비교 조건 | 모델 checkpoint, 평가 입력, 측정법 |
| 모델 구조 고정 | 모델을 해석하며 반복 결정할 필요가 없는 항목 | Graph, shape, operator, Tensor lifetime |
| Target 최적화 고정 | QRB2210 측정 결과로 선택한 실행 방법 | Offset, layout, tile, thread, backend |

고정한다고 해서 입력에 의존하는 Conv·TDNN·Pooling 계산 자체를 생략하는 것은 아니다. 런타임의 graph 해석, shape 추론, 메모리 할당, layout 결정, kernel 선택 같은 **준비와 판단 과정**을 생략하는 것이다.

## 3.4 단계별 Freeze Gate

| Freeze ID | 확정 단계 | 하나씩 고정하는 항목 | 이후 런타임에서 생략되는 것 |
| --- | --- | --- | --- |
| `F0-TARGET` | Phase 0 | QRB2210, AArch64, OS·driver 조건 | 다른 ISA·기기용 분기와 범용 backend 탐색 |
| `F0-MODEL` | Phase 0 | CAM++ checkpoint, 모델 checksum | 실행 시 모델 버전 판단 |
| `F0-EVAL` | Phase 0 | 1·3·10초 입력, trial list, 정확도 기준 | 실험마다 평가 조건을 다시 정하는 과정 |
| `F1-BASELINE` | Phase 1 | 측정 절차, warm-up, 반복 수, 기준 성능 | 결과 비교 방식의 변경 |
| `F2-GRAPH` | Phase 2 | ONNX opset, operator 목록, 상수, 실행 graph | ONNX graph parsing·constant folding |
| `F2-SHAPE` | Phase 2 | 입력 length bucket과 Tensor shape | 동적 shape inference와 크기 계산 |
| `F3-SEMANTICS` | Phase 3 | 각 operator의 정확한 계산 규칙 | 범용 operator 검증과 불필요한 기능 분기 |
| `F4-LIFETIME` | Phase 4 | Tensor first/last use, in-place 가능 여부 | 실행 중 Tensor 생존 기간 분석 |
| `F4-OFFSET` | Phase 4 | Arena 크기, Tensor offset, alignment, alias | Tensor별 `malloc/free`와 allocator 실행 |
| `F4-DENSE` | Phase 4 | Dense slab 크기와 layer별 channel offset | `concat` 할당과 전체 prefix 복사 |
| `F5-LAYOUT` | Phase 5 | `[T,C]` 형태, channel padding, weight block 순서 | 실행 중 transpose·weight repacking |
| `F5-KERNEL` | Phase 5 | Operator별 NEON kernel과 tile 크기 | CPU feature 확인과 kernel 후보 탐색 |
| `F5-THREAD` | Phase 5 | Operator별 1·2·4 thread 및 작업 분할 | 동적 thread 수 결정과 work scheduling |
| `F6-FUSION` | Phase 6 | Conv·Bias·Activation 등의 fusion plan | 중간 operator 호출과 Tensor read/write |
| `F7-QUANT` | Phase 7 | INT8 적용 구간, scale, zero point, accumulation dtype | 실행 중 calibration과 quantization 방식 선택 |
| `F7-PACK` | Phase 7 | QRB2210 NEON용 packed weights | 모델 로딩 시 weight 변환·패딩·패킹 |
| `F8-SCHEDULE` | Phase 8 | Operator 실행 순서와 dependency | 실행 graph 재구성과 dispatch plan 생성 |
| `F8-BINARY` | Phase 8 | Binary format, version, checksum | ONNX 파일 로딩과 모델 최적화 |
| `F9-LOAD` | Phase 9 | `mmap`·prefault·warm-up 정책 | 매 실행마다 로딩 전략 선택 |
| `F10-GPU` | Phase 10 | CPU/GPU 분할점, GPU arena offset, 동기화 위치 | 실행 시 graph partition과 backend 협상 |
| `F11-TUNE` | Phase 11 | 해당 보드의 최종 tile·thread·backend profile | 매 inference의 autotune |
| `F12-DEPLOY` | Phase 12 | 장시간 운용 가능한 clock·thread 정책 | 불안정하거나 throttling되는 설정 재탐색 |
| `F13-NOTE9` | Phase 13 | Exynos 전용 layout·packing·backend | Note9에서 QRB2210 설정을 잘못 재사용하는 것 |
| `F14-EXPAND` | Phase 14 | Whisper·Zipformer별 graph와 새 operator | 모델마다 공통 runtime 전체를 다시 설계하는 것 |

## 3.5 Freeze 승인 규칙

각 항목은 다음 조건을 모두 만족해야 고정한다.

- [ ] 관련 Experiment ID가 존재한다.
- [ ] Reference 결과와 정확도 기준을 통과한다.
- [ ] 동일 조건 반복 benchmark가 안정적이다.
- [ ] 선택한 값과 탈락한 후보가 기록되어 있다.
- [ ] 적용 대상 device·model checksum이 기록되어 있다.
- [ ] 고정 해제 조건이 기록되어 있다.

Freeze 기록 예시:

```text
Freeze ID: F4-OFFSET-v3
Device: QRB2210
Model checksum: 6f...
Evidence: MEM-007, MEM-009
Decision: arena_3s = 18.4MB
Reason: 정확도 동일, peak RSS 31% 감소
Reopen when: graph 변경 또는 10초 bucket 추가
```

다음 변경이 발생하면 관련 Freeze를 다시 검증한다.

- 모델 graph 또는 checkpoint 변경
- 입력 길이 bucket 변경
- Quantization scale 또는 dtype 변경
- Kernel layout 변경
- UNO Q OS·Mesa·GPU driver 변경
- 다른 CPU·GPU target으로 이동

---

# 4. 전체 연구 단계

각 Phase의 `종료 조건`은 동시에 해당 항목의 Freeze 조건이다. Phase가 끝나면 위 표의 Freeze ID를 발급하고, 다음 Phase에서는 그 값을 기본 고정값으로 사용한다.

## Phase 0. 연구 환경과 기준 고정

### 목표

모든 최적화 결과를 같은 조건에서 다시 측정할 수 있도록 기준을 고정한다.

### 수행 항목

- [ ] CAM++ 학습 checkpoint와 Git commit 고정
- [ ] ONNX opset과 export script 버전 고정
- [ ] 1초·3초·10초 기준 음성 세트 생성
- [ ] 정상·잡음·무음·짧은 음성 포함
- [ ] 정확도 평가용 고정 trial list 생성
- [ ] CPU governor와 보드 전원 조건 기록
- [ ] Cold run과 warm run 절차 분리
- [ ] 결과 JSON schema 정의

### 산출물

```text
benchmarks/campplus/
├── README.md
├── inputs/
├── reference_outputs/
├── trial_list.txt
└── benchmark_protocol.md
```

### 종료 조건

- 같은 binary를 같은 조건에서 반복 실행했을 때 결과와 성능이 재현된다.
- 기준 입력과 정답 산출물에 checksum이 기록되어 있다.

---

## Phase 1. 현재 CAM++ baseline 분석

### 목표

최적화 전에 현재 병목을 수치로 확인한다.

### 수행 항목

- [ ] PyTorch CPU single-thread 측정
- [ ] PyTorch CPU 4-thread 측정
- [ ] ONNX Runtime CPU 측정
- [ ] 가능한 경우 XNNPACK 또는 Arm backend 측정
- [ ] operator별 시간 측정
- [ ] operator별 입력·출력 shape 기록
- [ ] activation peak memory 계산
- [ ] 가중치와 activation의 예상 이동량 계산
- [ ] `concat`, transpose, reshape, quantize/dequantize 횟수 기록
- [ ] 1·3·10초 입력 길이별 profile 생성

### 우선 확인할 연산

- FCM의 Conv2D
- D-TDNN의 Linear·Conv1D
- Dense concat
- CAM의 global·segment pooling
- Sigmoid·elementwise multiply
- 최종 pooling과 embedding layer

### 산출물

- `baseline_report.md`
- `operator_profile.csv`
- `tensor_lifetime.csv`
- `graph.svg` 또는 `graph.json`

### 종료 조건

- 전체 시간의 80% 이상을 차지하는 operator 목록이 확인된다.
- peak activation과 모델 크기가 수치로 산출된다.
- 계산 병목과 메모리 병목에 대한 1차 가설이 작성된다.

---

## Phase 2. ONNX 공통 원본 확정

### 목표

PyTorch를 제거하기 위한 정확한 공통 모델 표현을 만든다.

### 수행 항목

- [ ] Eval mode에서 ONNX export
- [ ] Dropout·학습 전용 node 제거
- [ ] Constant folding
- [ ] Conv + BatchNorm folding
- [ ] 불필요한 Identity·Reshape·Transpose 제거
- [ ] 고정 길이 또는 입력 길이 bucket 결정
- [ ] ONNX checker 통과
- [ ] ONNX Runtime 출력과 PyTorch 출력 비교
- [ ] 전체 평가 세트에서 embedding과 EER 비교

### 포맷 원칙

- ONNX는 보존과 검증을 위한 공통 원본이다.
- ONNX 파일 자체를 최종 배포 binary로 간주하지 않는다.
- target별 layout·packing·offset은 ONNX와 분리한다.

### 종료 조건

- 모든 기준 입력에서 PyTorch와 ONNX 결과가 합격 기준을 통과한다.
- 사용되는 operator와 dtype이 완전히 목록화되어 있다.

---

## Phase 3. 최소 C/C++ FP32 runtime

### 목표

CAM++에 필요한 operator만 구현한 PyTorch 없는 runtime을 만든다.

### 공통 Tensor 정보

```text
shape
dtype
stride
arena offset
quantization scale
backend location
first use / last use
```

### 우선 구현 operator

- [ ] Conv1D
- [ ] Conv2D
- [ ] Linear/MatMul
- [ ] BatchNorm이 fold되지 못한 normalization
- [ ] ReLU·Sigmoid 등 activation
- [ ] Average pooling·statistics pooling
- [ ] View·reshape
- [ ] Elementwise add·multiply
- [ ] CAM++ 전용 dense slab view

### 원칙

- 범용 ONNX Runtime 전체를 재구현하지 않는다.
- CAM++ graph에 실제로 존재하는 operator만 구현한다.
- 첫 구현은 단순한 FP32 reference kernel로 작성한다.
- 각 operator는 ONNX Runtime 결과와 단독 비교할 수 있어야 한다.

### 종료 조건

- 전체 CAM++ graph가 C/C++ FP32로 끝까지 실행된다.
- 모든 operator unit test와 end-to-end 정확도 검증을 통과한다.

---

## Phase 4. 정적 offset 기반 Tensor Arena

### 목표

Tensor별 동적 할당과 불필요한 activation 복사를 제거한다.

### 오프라인 memory planning

1. Graph를 topological order로 정렬한다.
2. 각 Tensor의 `first_use`와 `last_use`를 계산한다.
3. 필요한 alignment를 계산한다.
4. 수명이 겹치지 않는 Tensor에 같은 offset을 배정한다.
5. 입력 길이 bucket별 arena plan을 생성한다.
6. 최종 offset과 arena 크기를 binary에 기록한다.

### 기본 정렬 규칙

```text
Tensor 시작       64-byte 정렬
가중치 section    4096-byte 정렬
NEON INT8 block   input channel 16개 단위
output block      4개 또는 8개 단위
코어별 scratch    서로 다른 cache line
```

### Tensor descriptor 예시

```cpp
struct TensorDesc {
    uint32_t offset;
    uint32_t size;
    uint16_t shape[4];
    uint8_t dtype;
    uint8_t flags;
};

uint8_t* tensor_ptr(uint8_t* arena, const TensorDesc& tensor) {
    return arena + tensor.offset;
}
```

### CAM++ dense connection 전용 설계

기존 방식:

```text
concat(old_0, old_1, ..., new)
→ 새 Tensor 할당
→ 전체 prefix 복사
```

변경 방식:

```text
dense_slab = [T, C_max_padded]

layer 0 output → channel offset 0
layer 1 output → 다음 빈 channel offset
layer 2 output → 다음 빈 channel offset
```

이전 출력은 복사하지 않고 prefix view로 참조한다.

### 종료 조건

- 추론 중 Tensor allocation 0회
- Dense concat에 의한 전체 Tensor 복사 0회
- 입력 길이 bucket별 arena 크기가 기록됨
- offset alias가 정확도나 메모리 손상을 일으키지 않음

---

## Phase 5. CPU NEON FP32 최적화

### 목표

메모리 구조가 고정된 FP32 runtime에 QRB2210용 NEON kernel을 적용한다.

### 최적화 순서

1. 가장 많은 시간을 사용하는 operator부터 변경한다.
2. Scalar reference와 NEON 결과를 layer별 비교한다.
3. 연산 하나씩 변경하고 benchmark를 남긴다.
4. tile 크기와 thread 수를 동시에 바꾸지 않는다.

### 적용 항목

- [ ] 채널 4·8·16 단위 패딩
- [ ] Direct Conv1D 또는 implicit GEMM
- [ ] `im2col` activation 복사 제거
- [ ] FCM Conv2D tiling
- [ ] Linear·1×1 Conv microkernel
- [ ] 연속 weight access
- [ ] software prefetch 거리 실험
- [ ] 코어별 scratch 분리
- [ ] 1·2·4 thread 비교
- [ ] static work partition
- [ ] barrier 횟수 최소화

### Weight layout 예시

```text
[output block][kernel position][input block][output lane][input lane]
```

가중치는 실행 순서대로 배치한다.

```text
Layer 0 packed weights
Layer 1 packed weights
Layer 2 packed weights
...
```

### 종료 조건

- 모든 NEON kernel이 reference test를 통과한다.
- 단일 최적화별 개선량이 기록되어 있다.
- operator별 최적 tile과 thread 수가 확정된다.

---

## Phase 6. 연산 융합

### 목표

중간 Tensor의 RAM write와 다음 연산의 read를 제거한다.

### 융합 후보

- [ ] Conv + BatchNorm — 가능하면 오프라인 weight folding
- [ ] Conv + Bias + ReLU
- [ ] Linear + Bias + Activation
- [ ] Pooling + Scale + Sigmoid
- [ ] Mask generation + elementwise multiply
- [ ] Quantize + Conv/Linear + Requantize
- [ ] View·reshape의 실제 copy 제거

### 검증 항목

- 융합 전후 수치 차이
- 제거된 Tensor 수
- 줄어든 arena 크기
- 줄어든 DRAM read/write 추정량
- latency 변화

### 종료 조건

- 융합별 정확도와 성능이 개별 기록된다.
- 융합 후에도 fallback reference path가 유지된다.

---

## Phase 7. INT8 양자화와 가중치 사전 패킹

### 목표

가중치 크기와 메모리 대역폭 사용량을 줄이고 NEON 처리 효율을 높인다.

### 단계 A — 제한적 INT8

- [ ] Linear·Conv 계열 weight를 per-output-channel INT8로 변환
- [ ] 입력 activation 동적 또는 calibration 기반 양자화
- [ ] INT32 accumulation
- [ ] 출력은 FP32로 복원
- [ ] Pooling·Sigmoid·민감한 연산은 FP32 유지

### 단계 B — INT8 subgraph

- [ ] 연속된 Conv·Linear 구간을 W8A8로 유지
- [ ] 중간 Quantize/Dequantize 제거
- [ ] Fused requantization 적용
- [ ] layer별 saturation과 scale 범위 기록

### 오프라인 변환

```text
ONNX weights
→ BatchNorm folding
→ INT8 quantization
→ channel padding
→ NEON block packing
→ packed weight section 저장
```

### 주의점

- A53 계열에는 최신 `SDOT` 명령이 없을 가능성이 있으므로 실제 CPU feature를 확인한다.
- INT8이 항상 계산 속도를 높이지는 않지만 weight traffic 감소 효과는 별도로 측정한다.
- 정확도 비교 없이 latency만으로 채택하지 않는다.

### 종료 조건

- INT8 모델이 정확도 기준을 통과한다.
- FP32 대비 모델 크기, latency, RSS, EER 변화가 표로 정리된다.

---

## Phase 8. QRB2210 전용 모델 컴파일러와 binary

### 목표

그래프, offset, 스케줄, quantization, packed weights를 하나의 배포 산출물로 만든다.

### 컴파일 과정

```text
ONNX
→ graph canonicalization
→ constant folding
→ operator fusion
→ shape inference
→ quantization
→ target layout 결정
→ weight packing
→ tensor lifetime 분석
→ arena offset 배정
→ kernel 선택
→ thread schedule 생성
→ target binary 직렬화
```

### Binary 구조

```text
campplus_qrb2210_int8.bin
├── header
│   ├── magic
│   ├── format version
│   ├── target architecture
│   ├── dtype
│   └── checksum
├── operator table
├── tensor descriptor table
├── execution schedule
├── arena plans
├── quantization parameters
└── packed weights
```

### Target 분리

```text
model.onnx
campplus_qrb2210_int8.bin
campplus_exynos9810_int8.bin
```

공통 ONNX에서 target별 binary를 별도로 생성한다. Cache·driver·지원 명령이 다른 기기에 같은 packed binary를 강제로 사용하지 않는다.

### 종료 조건

- 보드에서 ONNX나 PyTorch 없이 binary 하나로 실행된다.
- binary format version과 checksum 검증이 동작한다.
- 잘못된 target binary 로딩을 거부한다.

---

## Phase 9. 모델 로딩과 cold-start 최적화

### 목표

모델 로딩 이후 page fault와 런타임 변환을 최소화한다.

### 수행 항목

- [ ] Packed binary `mmap`
- [ ] `MAP_POPULATE` 지원 여부 확인
- [ ] `madvise(MADV_WILLNEED)` 비교
- [ ] 시작 시 weight page prefault
- [ ] `mlock` 가능 여부와 효과 측정
- [ ] weight section read-only 보호
- [ ] arena 한 번 할당
- [ ] 고정 worker thread 시작
- [ ] warm-up inference 실행

### 원칙

- `/dev/mem`으로 물리 주소를 직접 건드리지 않는다.
- 모델이 작으면 복잡한 huge page 최적화보다 단순한 prefault를 먼저 검증한다.
- Cold start와 warm inference를 분리해 보고한다.

### 종료 조건

- 첫 실행과 warm 실행의 차이가 설명 가능하다.
- 모델 로딩 중 수행되는 모든 변환이 목록화되어 있다.

---

## Phase 10. GPU 가속 실험

### 목표

Adreno 702에서 GPU가 실제로 이득인 큰 서브그래프만 찾는다.

### 시작 전 capability 확인

```bash
clinfo
vulkaninfo --summary
```

확인할 항목:

- Unified memory 보고 여부
- OpenCL SVM capability
- GPU local/global memory 정보
- 최대 work-group 크기
- FP16 지원
- Vulkan storage buffer alignment
- Vulkan/OpenCL extension 목록

### 비교할 실행 배치

1. 전체 CPU
2. FCM만 GPU, D-TDNN과 pooling은 CPU
3. FCM + 큰 Linear/Conv1D만 GPU
4. 전체 모델 GPU

### GPU memory 원칙

```text
하나의 큰 GPU arena
Tensor A → offset A
Tensor B → offset B
Tensor C → offset C
```

- GPU 중간 Tensor는 서브그래프가 끝날 때까지 GPU에 유지한다.
- CPU/GPU 경계에서만 동기화한다.
- kernel마다 CPU로 결과를 회수하지 않는다.
- Zero-copy와 host-visible memory가 항상 빠르다고 가정하지 않는다.
- 한 번의 작은 copy가 느린 공유 mapping보다 빠른지도 비교한다.

### 예상 배치

| 연산 | 우선 후보 |
| --- | --- |
| FBank | CPU NEON |
| 큰 FCM Conv2D | GPU 후보 |
| 짧고 좁은 TDNN | CPU 후보 |
| 큰 Linear·Conv1D | CPU/GPU 실측 |
| CAM·Pooling·Sigmoid | producer와 융합 |
| 최종 embedding | CPU |

### 종료 조건

- GPU가 이득인 최소 Tensor 크기 또는 시퀀스 길이가 확인된다.
- CPU/GPU 경계별 전달 바이트와 동기화 시간이 측정된다.
- GPU 사용 여부가 추측이 아니라 benchmark로 결정된다.

---

## Phase 11. Autotuner

### 목표

모델과 기기에 맞는 tile·thread·backend를 자동 선택한다.

### 튜닝 후보

- Conv·GEMM tile 크기
- Prefetch 거리
- Operator별 1·2·4 thread
- CPU/GPU 배치
- 입력 길이 bucket
- FP32·INT8 kernel 선택

### 결과 저장 예시

```json
{
  "device": "qrb2210",
  "model_checksum": "...",
  "operators": {
    "fcm_conv_0": {"backend": "gpu", "tile": "8x16"},
    "tdnn_12": {"backend": "cpu", "threads": 2},
    "linear_0": {"backend": "cpu", "threads": 4}
  }
}
```

### 종료 조건

- 첫 설치 또는 모델 변경 때만 autotune한다.
- 동일 model checksum에서는 저장된 plan을 재사용한다.
- autotune 실패 시 안전한 CPU fallback이 동작한다.

---

## Phase 12. 장시간 안정성·열·전력 검증

### 목표

짧은 benchmark에서 빠른 설정이 실제 연속 실행에서도 유지되는지 확인한다.

### 수행 항목

- [ ] 30분·1시간 반복 inference
- [ ] CPU/GPU 온도 기록
- [ ] clock 변화 기록
- [ ] p95·p99 latency 변화 기록
- [ ] RSS 증가 여부 확인
- [ ] 오류·NaN·embedding drift 확인
- [ ] CPU-only와 CPU/GPU 전력 비교

### 종료 조건

- thermal throttling 이후에도 요구 latency를 만족한다.
- 메모리 누수와 누적 정확도 오류가 없다.

---

## Phase 13. Galaxy Note9 호환성 검증

### 목표

공통 AArch64 계층의 재사용 범위와 target별 최적화 차이를 확인한다.

### 대상

- Exynos 9810 CPU
- Mali-G72 MP18 GPU
- LPDDR4X shared memory
- Android runtime 제약

### 전략

- 공통 ONNX와 operator semantics 재사용
- 공통 Tensor·Arena 설계 재사용
- AArch64 NEON reference kernel 재사용
- Exynos cache와 CPU feature에 맞게 다시 패킹
- Vulkan backend는 별도 capability 확인
- NNAPI는 장기 핵심 backend가 아니라 비교용으로 사용

### 산출물

```text
campplus_exynos9810_int8.bin
note9_benchmark_report.md
qrb2210_vs_exynos9810.csv
```

---

## Phase 14. Whisper encoder와 Zipformer 확장

### 순서

1. Whisper encoder의 ONNX/CT2 baseline 측정
2. CAM++에서 만든 Tensor·Arena·binary infrastructure 재사용
3. Conv1D·Linear·normalization kernel 재사용
4. Attention·Softmax·LayerNorm 추가
5. KV cache가 필요한 구조라면 별도 persistent arena 설계
6. Zipformer의 multi-rate sequence와 downsampling에 맞는 plan 추가

### CT2의 역할

- Whisper는 CTranslate2 공식 지원 대상이므로 성능 기준선으로 사용한다.
- CAM++와 Zipformer의 공통 최종 포맷으로 CT2를 사용하지 않는다.
- 자체 runtime이 CT2보다 빠른지보다, 요구 기능·메모리·재사용성을 함께 비교한다.

---

# 5. 실행 엔진 구조

## 5.1 컴포넌트

```text
model_compiler/
├── onnx_loader
├── graph_optimizer
├── quantizer
├── layout_planner
├── weight_packer
├── memory_planner
├── scheduler
└── serializer

runtime/
├── model_loader
├── tensor_arena
├── thread_pool
├── dispatcher
├── cpu_aarch64
├── gpu_opencl
├── gpu_vulkan
└── profiler
```

## 5.2 Backend 우선순위

| Backend | 역할 | 우선순위 |
| --- | --- | --- |
| `cpu_reference` | 정확도 기준 | 필수 |
| `cpu_aarch64_fp32` | 첫 최적화 baseline | 필수 |
| `cpu_aarch64_int8` | QRB2210 주력 | 필수 |
| `gpu_opencl` | UNO Q 초기 GPU 검증 | 실험 |
| `gpu_vulkan` | Adreno·Mali 공통화 | 후속 |
| `android_nnapi` | Note9 비교 | 선택 |
| `qualcomm_qnn` | 지원 확인 후 | 보류 |
| `mcu_cmsis` | STM32 Wakeword·VAD | 별도 프로젝트 |

---

# 6. 벤치마크 프로토콜

## 6.1 입력 조건

| 입력 | 목적 |
| --- | --- |
| 1초 음성 | 짧은 입력에서 호출·동기화 오버헤드 확인 |
| 3초 음성 | 일반 speaker embedding 기준 |
| 10초 음성 | 긴 sequence에서 GPU 이점 확인 |
| 무음 | 수치 안정성과 전처리 확인 |
| 잡음 음성 | 양자화 정확도 확인 |

## 6.2 반복 절차

1. 프로세스를 새로 시작해 cold start 측정
2. 20회 warm-up
3. 최소 100회 반복
4. p50·p95·p99 기록
5. RSS와 온도 기록
6. 출력 embedding checksum과 cosine 비교
7. thread·backend·dtype 변경 시 새 Experiment ID 발급

## 6.3 비교 표 템플릿

| Experiment | Backend | Dtype | Threads | Input | p50 | p95 | RTF | RSS | Arena | EER | 비고 |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BASE-001 | ONNX CPU | FP32 | 1 | 3s |  |  |  |  |  |  |  |
| CPU-001 | Custom CPU | FP32 | 1 | 3s |  |  |  |  |  |  |  |
| CPU-002 | Custom CPU | INT8 | 4 | 3s |  |  |  |  |  |  |  |
| GPU-001 | FCM GPU | FP16/32 |  | 3s |  |  |  |  |  |  |  |

---

# 7. 노션 데이터베이스 설계

## 7.1 Experiments DB

| 속성 | 타입 | 설명 |
| --- | --- | --- |
| Experiment ID | Title | `CPU-001`, `MEM-003` 형식 |
| Status | Select | Planned / Running / Valid / Rejected |
| Date | Date | 실행일 |
| Git SHA | Text | 코드 버전 |
| Model checksum | Text | 모델 버전 |
| Device | Select | QRB2210 / Exynos9810 / STM32U585 |
| Backend | Select | ONNX / CPU FP32 / CPU INT8 / OpenCL / Vulkan |
| Input length | Number | 초 단위 |
| Threads | Number | CPU thread 수 |
| Optimization | Multi-select | Arena / Packing / Fusion / Tiling / Quantization |
| p50 / p95 | Number | 지연시간 |
| RTF | Number | Real-time factor |
| Peak RSS | Number | MB |
| Arena | Number | MB |
| EER / MinDCF | Number | 정확도 |
| Temperature | Number | 최고 온도 |
| Conclusion | Text | 한 문장 결론 |
| Artifact | URL/Text | 결과 파일 위치 |

## 7.2 Decision Log DB

| 속성 | 설명 |
| --- | --- |
| Decision | 결정 내용 |
| Date | 결정일 |
| Evidence | 관련 Experiment ID |
| Alternatives | 검토한 대안 |
| Reason | 선택 이유 |
| Revisit condition | 결정을 다시 검토할 조건 |

## 7.3 Risk Register DB

| 위험 | 영향 | 가능성 | 대응 |
| --- | --- | --- | --- |
| INT8 정확도 하락 | 높음 | 중간 | 민감 연산 FP32 유지, calibration 재설계 |
| GPU 호출 비용이 이득 상쇄 | 중간 | 높음 | 큰 서브그래프만 dispatch |
| Linux driver capability 부족 | 높음 | 중간 | CPU path를 주력 fallback으로 유지 |
| Thermal throttling | 높음 | 중간 | 장시간 benchmark와 adaptive thread |
| 과도한 범용 runtime 개발 | 높음 | 중간 | CAM++ 필수 operator만 구현 |
| Target별 packed binary 증가 | 낮음 | 높음 | 공통 ONNX + 자동 target build |
| Offset alias 버그 | 높음 | 중간 | Guard region·ASan·reference test |

---

# 8. 실험 기록 템플릿

## Experiment ID — 제목

### 가설

> 무엇을 바꾸면 어떤 지표가 왜 개선될 것으로 예상하는가?

### 변경 사항

- 변경한 코드:
- 변경한 모델:
- 변경한 layout/offset:
- 이전 실험과의 차이:

### 고정 조건

- Device:
- Git SHA:
- Model checksum:
- Input set:
- CPU governor:
- Temperature start:

### 결과

| 지표 | Before | After | 변화 |
| --- | ---: | ---: | ---: |
| p50 latency |  |  |  |
| p95 latency |  |  |  |
| Peak RSS |  |  |  |
| Arena size |  |  |  |
| EER |  |  |  |

### 해석

- 계산량 감소:
- 메모리 이동 감소:
- 예상과 달랐던 점:
- 오차 또는 측정 한계:

### 결론

- [ ] 채택
- [ ] 기각
- [ ] 추가 실험 필요

### 다음 실험

- 

---

# 9. Git과 산출물 관리

## 9.1 저장소에 포함

```text
src/
compiler/
runtime/
tests/
benchmarks/
experiments/
configs/
scripts/
docs/
```

- 모델 컴파일러와 runtime 코드
- Operator unit test
- Benchmark harness
- 실험 config
- 작은 결과 CSV와 report
- Binary format 문서

## 9.2 Git에서 제외

```gitignore
datasets/
models/
runs/
checkpoints/
*.pt
*.pth
*.onnx
*.bin
```

배포 binary가 작고 버전 관리가 필요하면 Git LFS 또는 release artifact로 관리한다. 결과의 근거가 되는 checksum과 생성 명령은 Git에 남긴다.

## 9.3 권장 구조

```text
camp-runtime/
├── README.md
├── compiler/
├── runtime/
│   ├── core/
│   └── backends/
├── tests/
├── benchmarks/
│   └── qrb2210/
├── experiments/
│   └── campplus/
├── configs/
├── scripts/
└── docs/
    ├── binary_format.md
    ├── memory_planner.md
    └── kernel_design.md
```

---

# 10. 구현 우선순위

## P0 — 반드시 먼저

- [ ] 기준 CAM++ ONNX 확정
- [ ] 고정 benchmark input과 reference output 생성
- [ ] Operator·Tensor profile 생성
- [ ] Scalar C/C++ reference runtime
- [ ] Tensor lifetime 분석
- [ ] Dense concat 제거
- [ ] 정적 Tensor Arena

## P1 — CPU 가속 핵심

- [ ] Conv1D·Linear NEON
- [ ] FCM Conv2D tiling
- [ ] 가중치 사전 패킹
- [ ] 1·2·4 thread schedule
- [ ] Operator fusion
- [ ] INT8 정확도 검증

## P2 — 시스템 최적화

- [ ] Packed binary format
- [ ] `mmap`·prefault
- [ ] Autotuner
- [ ] 장시간 thermal test
- [ ] Native service 배포

## P3 — 확장

- [ ] Adreno GPU 실험
- [ ] Note9 Vulkan/NEON 검증
- [ ] Whisper encoder
- [ ] Zipformer
- [ ] STM32 Wakeword·VAD runtime

---

# 11. 바로 시작할 작업

1. 기준 CAM++ 모델과 1·3·10초 입력을 고정한다.
2. PyTorch와 ONNX Runtime에서 operator별 profile을 뽑는다.
3. ONNX graph의 모든 operator와 Tensor shape를 CSV로 만든다.
4. Dense block마다 최대 채널 수와 Tensor lifetime을 계산한다.
5. Offset이 없는 단순 FP32 C++ reference runtime부터 만든다.
6. Reference가 맞으면 정적 arena와 dense slab을 적용한다.
7. 그다음 가장 느린 Conv/Linear 하나만 NEON으로 교체한다.
8. 모든 변경은 별도 Experiment ID로 benchmark한다.

첫 번째 연구 milestone은 GPU가 아니라 다음 상태다.

> **PyTorch 없이 CAM++ FP32가 QRB2210에서 실행되고, 추론 중 동적 할당과 dense concat 복사가 없는 상태**

이 milestone이 완성된 후 NEON, INT8, GPU를 순서대로 적용한다.

---

# 12. 참고 자료

- [Arduino UNO Q 공식 데이터시트](https://docs.arduino.cc/resources/datasheets/ABX00162-datasheet.pdf)
- [Qualcomm QRB2210 공식 제품 정보](https://www.qualcomm.com/internet-of-things/products/q2-series/qrb2210)
- [Qualcomm QRB2210 데이터시트](https://docs.qualcomm.com/bundle/publicresource/80-30843-1.pdf)
- [CAM++ 논문](https://arxiv.org/abs/2303.00332)
- [CTranslate2 지원 모델](https://opennmt.net/CTranslate2/guides/transformers.html)
- [CTranslate2 Quantization](https://opennmt.net/CTranslate2/quantization.html)
- [ONNX Runtime Graph Optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
- [Khronos OpenCL Specification](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html)
- [Samsung Exynos 9810](https://semiconductor.samsung.com/processor/mobile-processor/exynos-9-series-9810/)
- [Android NNAPI 안내](https://developer.android.com/ndk/guides/neuralnetworks)
