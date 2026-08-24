# campp_sv_498

bucket 498 전용 배포 단위다.  오디오 5.0초 (498 frame) 입력을
192차원 화자 임베딩으로 바꾼다.

## 구성

| 파일 | 내용 |
|---|---|
| `campp_sv_498.camppmodel` | execution plan, packed weights, streaming weights/schedule, kernel dispatch table, frontend/postprocess 계약 |
| `campp_runtime` | 런타임 바이너리.  **v5/v4 NEON 커널 구현체가 여기 있다** |
| `deployment.json` | 모델과 런타임의 sha256, 호환 계약 |

**모델과 런타임은 함께 배포해야 한다.**  모델은 "어느 operator에 어느 커널을
쓸지"를 담고, 런타임은 "그 커널이 무엇인지"를 담는다.  짝이 어긋나면 조용히 다른
커널이 돌기 때문에 `deployment.json`의 sha256으로 대조한다.

## 모델에 들어 있는 최적화

- INT8 양자화
- dense slab (Concat 52개 제거)
- o4i4 weight 재배치 -- 런타임 재패킹 불필요
- E7 operator fusion (1278 -> 884 op)
- tensor arena 배치
- layer-hybrid kernel dispatch 표: {"fused_combined_fixed": 1, "fused_combined_v5": 55, "qconv_mac_fixed": 2, "qconv_v5": 113}
- weight streaming schedule (RAM 절감용)

## 모델에 없는 것

- v5/v4 NEON 커널 구현체 (런타임 바이너리)
- 화자 검증 임계값 (보정 전)
- 오디오 프론트엔드 구현 (계약만 담김)

## 실행

```bash
./campp_runtime --plan <plan.bin> --weights <weights.bin> \
    --input <feature.f32> --audio-seconds 5.0 \
    --warmup 5 --repeat 30 --threads 1
```

현재 런타임은 분리된 plan/weights 파일을 받는다.  `.camppmodel`을 직접 읽는 경로는
C 로더 확장이 끝나야 쓸 수 있다 -- `deployment.json`의 `runtime_reads_model_package`
가 그 상태를 알려준다.

입력은 FBank 498x80 float32다.  파라미터는 모델의 frontend 계약을 따른다
(80 mel, 25 ms/10 ms, dither 0, hamming, time축 CMVN).
