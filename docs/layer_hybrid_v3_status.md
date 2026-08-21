# Layer-hybrid V3 진행 상태 (2026-08-20, 최종)

## 지금 상태 한 줄

**4개 bucket(98/298/498/998) 전부** operator 단위 latency 실측으로 선택해 C 소스에
반영, 빌드, 검증까지 마쳤다.  정확도 gate는 전부 통과했고 V2 대비 6% 내외 개선이다.
다만 **ORT와 비교하면 RTF가 2.1~2.2배 느리다** -- 아래 "ORT 대비 최종 비교" 참고.

## 측정 결과

| bucket | ordinary v4 | ordinary v5 | 개선 | fused hybrid | fused v5 | fused winner |
|---|---|---|---|---|---|---|
| 98  | 167.9 ms | 155.0 ms | +7.66% | 139.0 ms | 153.3 ms | combined_hybrid |
| 298 | 489.1 ms | 441.4 ms | +9.75% | 402.3 ms | 456.1 ms | combined_hybrid |
| 498 | 804.4 ms | 725.4 ms | +9.82% | 671.9 ms | 763.3 ms | combined_hybrid |
| 998 | 1642.6 ms | 1512.7 ms | +7.91% | 1333.9 ms | 1535.8 ms | combined_hybrid |

ordinary는 4개 bucket 모두 v5가 승자다.  개선폭은 498에서 정점(9.82%)이고 998에서
7.91%로 내려온다.  fused는 전체로는 hybrid가 이기지만 **op 단위로는 절반가량이
combined_v5로 넘어간다** — layer-hybrid가 의미 있는 이유다.

## C 소스에 반영된 테이블

`src/c/profill/optimization/candidates/conv_layer_hybrid/conv_layer_hybrid_plan.c`

| bucket | qconv mac_fixed / v5 | fused fixed / v5 |
|---|---|---|
| 98  | 2 / 112 | 5 / 55 |
| 298 | 2 / 113 | 4 / 55 |
| 498 | 2 / 113 | 1 / 55 |
| 998 | 2 / 111 | 6 / 55 |

테이블에 없는 op는 incumbent(v4 / combined_hybrid)로 간다.  선택 규칙은 incumbent
대비 1% 이상 빠를 때만 교체다.  4개 bucket 전부 실측 반영이 끝났으므로 fallback으로
가는 bucket은 없다.

빌드 산출물: `build/profill/final_v3_hybrid/` — **원본 `05_build_final_v3.sh`로
빌드된 것이어야 한다** (`build.txt`에 `object_cache` 항목이 없으면 원본).
`test_final_candidate_suite` 통과.

## 핵심 도구 (이번에 새로 만든 것)

`scripts/4_profill/optimization/18_remap_bucket_profile.py`
: 98 operator profile을 **weight tensor id 기준**으로 다른 bucket에 재매핑한다.
  operator_id는 bucket마다 다르지만(98↔298 겹침 68/225) weight tid는 4개 bucket
  전부 225/225 동일하다. winner를 옮기는 것이 아니라 **무엇을 측정할지**만 옮긴다.

`scripts/4_profill/optimization/19_generate_measured_v3_source.py`
: 측정이 끝난 bucket만으로 `conv_layer_hybrid_plan.c`를 생성한다. 인자로 bucket을
  나열한다 (`... 98 298 498`). 측정하지 않은 bucket은 테이블을 만들지 않아
  fallback이 보장된다.

## 남은 작업

**`16_select_multibucket_v3.py`의 `DEFAULT_PROFILE`이 98 단일 경로다.** bucket별
profile을 쓰도록 고치고, preflight에 "profile의 operator_id가 대상 plan에서
유효한가" 검사를 넣어야 한다.  이게 없어서 처음에 98 profile로 298을 측정할 뻔했다
(115개 중 79개가 QLinearConv조차 아닌 연산을 가리켰다).

**`tests/test_freeze_canonical.py` 실패.** `scripts/graph/`를 찾는데 실제 경로는
`scripts/2_graph/`다.  08-06자 파일로 이번 작업과 무관한 기존 문제다.

**공식 승격 판단이 미결이다.**  정확도 gate는 통과했지만 ORT 대비 RTF가 2배 느리다.
아래 "판단" 절 참고.

## ORT 대조 (참고)

별도로 진행한 ONNX vs C runtime 비교 결과다.

- 최초 발산 tensor: `cam_layer/ReduceMean` (op 43), float32 누적 순서 차이 ~1e-07
- 그 오차가 양자화 경계에서 1 LSB 뒤집힘 → CAM 곱셈 마스킹으로 증폭
- 15화자 중 4명에서 embedding cosine 0.99대 (나머지는 0.9999999999)
- **V2와 V3가 bitwise 동일** → V3 회귀가 아니라 C runtime 계열 고유 특성
- **EER 영향 없음**: bucket 298에서 ORT 5.0000% vs C 5.0000% (780 trial)
- 결론: ReduceMean을 float64로 바꾸는 수정은 권하지 않는다. 정확도 이득이 측정되지
  않는데 V2↔V3 bitwise gate만 깨진다.


---

# ORT 대비 최종 비교 (2026-08-20, governor=performance 2016MHz 고정)

여기가 결론이다.  **V3가 baseline이므로 비교 대상은 V2가 아니라 ONNX Runtime이다.**

측정 조건: 1 thread, `taskset -c 0`, warmup 5 / repeat 30, speaker_0000.
스크립트: `scripts/5_model/06_compare_ort_vs_v3.py` (+ `06_ort_runner.py`).
결과: `results/model_evaluation/final_v3/ort_vs_v3_rtf_ram.json`

## RTF -- ORT가 2배 이상 빠르다

| bucket | 오디오 | V3 RTF | ORT t1 RTF | V3/ORT |
|---|---|---|---|---|
| 98  | 1초  | 0.3279 | 0.1919 | 0.59x |
| 298 | 3초  | 0.3784 | 0.1773 | 0.47x |
| 498 | 5초  | 0.3716 | 0.1735 | 0.47x |
| 998 | 10초 | 0.3768 | 0.1723 | 0.46x |

bucket이 길수록 격차가 벌어진다.  layer-hybrid로 얻은 6%로는 2.2배 격차를 메울 수
없다.  다만 **둘 다 실시간 요건(RTF<1)은 충족**한다.

ORT 4-thread 수치도 JSON에 있지만 taskset으로 코어 0에 묶은 상태라 컨텍스트 스위칭만
발생해 1-thread보다 3.6배 느리다.  ORT의 실제 멀티스레드 성능이 아니므로 인용하지 말 것.

## Peak RSS -- V3가 압도적으로 유리하다

| bucket | V3 | ORT t1 | V3 절감 |
|---|---|---|---|
| 98  | 11.4 MB | 83.9 MB  | 86.4% |
| 298 | 14.6 MB | 91.9 MB  | 84.1% |
| 498 | 17.9 MB | 108.0 MB | 83.5% |
| 998 | 25.9 MB | 135.0 MB | 80.8% |

## 모델 로드

V3 18.6~19.3 ms vs ORT 3122~3621 ms -- **약 170배** 차이.  콜드 스타트가 잦은
워크로드에서는 이 항목이 지배적이다.

## 판단

| 항목 | 우위 | 배수 |
|---|---|---|
| RTF | ORT | 2.1~2.2x |
| Peak RSS | V3 | 3.6~7.4x |
| 모델 로드 | V3 | ~170x |

승격 판단은 배포 제약에 달렸다.  RAM이 빠듯하거나(UNO Q 3.6GB) 콜드 스타트가 잦으면
V3, 순수 처리량이면 ORT.  속도만으로는 V3를 정당화할 수 없다.

## 정확도 gate (전부 통과)

| bucket | V2=V3 bitwise | ORT cosine min | ORT max_abs |
|---|---|---|---|
| 98  | O | 0.9976820831 | 1.947e-01 |
| 298 | O | 0.9986209179 | 9.835e-02 |
| 498 | O | 0.9985132519 | 9.868e-02 |
| 998 | O | 0.9986981740 | 7.426e-02 |

NaN/Inf 없음, EER 차이 0.00%p (bucket 298, 780 trial).
`results/model_evaluation/final_v3/v3_vs_ort_by_bucket.json`

---

# 환경 상태 (인수인계)

**CPU governor를 performance로 고정해 두었다** (4코어 전부 2016MHz).
**재부팅하면 schedutil로 돌아간다.**  측정 재개 전에 확인할 것.

```bash
for c in 0 1 2 3; do cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor; done
# 다시 고정하려면 (sudo 비밀번호 필요)
for c in 0 1 2 3; do echo performance | sudo tee \
  /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor; done
```

schedutil 상태에서 잰 값은 절대값이 최대 2배까지 부풀고 bucket 간 순서가 뒤집히기도
했다(498이 298보다 빠르게 나옴).  개선율(비율)은 유지되지만 절대 RTF를 인용하려면
performance 고정이 필수다.

# 이번에 새로 만든 도구

| 파일 | 용도 |
|---|---|
| `scripts/4_profill/optimization/18_remap_bucket_profile.py` | 98 profile을 weight tid 기준으로 다른 bucket에 재매핑 |
| `scripts/4_profill/optimization/19_generate_measured_v3_source.py` | 측정 완료 bucket만으로 dispatch 소스 생성 |
| `scripts/4_profill/07_build_final_v3_cached.sh` | 오브젝트 캐시 빌드 -- **latency 측정 금지**, bitwise 검증 전용 |
| `scripts/5_model/05_verify_v3_vs_ort.py` | bucket별 V2 bitwise / ORT tolerance / E2E |
| `scripts/5_model/06_compare_ort_vs_v3.py` | ORT vs V3 RTF / RAM 비교 |
| `scripts/5_model/06_ort_runner.py` | ORT 단독 실행기 (peak RSS 격리 측정용) |

# 이번에 데인 것 (반복 방지)

**캐시 빌드는 성능이 다르다.**  `-flto` 때문에 소스를 한 번에 넘기는 것과 `-c`로
TU별 컴파일 후 링크하는 것의 인라이닝 결정이 다르다.  실측으로 bucket 98에서
원본 6.54% vs 캐시 1.56%였다.  스크립트 상단에 경고를 달아 뒀다.

**/tmp는 RAM 기반 tmpfs(1.8G)다.**  bucket 998 dump가 194MB라 tmpfs에 쓰면 RAM을
잠식해 실패한다.  검증 스크립트 기본 tmpdir을 `build/.verify_tmp`(디스크)로 바꿨다.

**측정 중 다른 명령을 돌리면 안 된다.**  진단용으로 단일 bucket을 돌렸다가 진행 중인
측정과 CPU를 경합해 298이 -31.78%로 나온 적이 있다.  측정 전에 `pgrep -af campp_`로
잔여 프로세스를 반드시 확인할 것.
