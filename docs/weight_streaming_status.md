# Weight streaming (weight RAM 최적화) 진행 상태 — 2026-08-21

## 한 줄 요약

4개 bucket(98/298/498/998) 전부 weight windowing을 적용해 **peak RSS를 5.42~5.52 MB,
정상 상태 RSS를 7.52~7.54 MB 줄였다.** 근본 원인은 Transparent Huge Page였고 고쳤다.
남은 것은 Official 측정(p50 gate 확정)과 bucket별 bitwise 검증이다.

## 근본 원인 — Transparent Huge Page

THP가 `[always]`인 커널에서 weights 파일 매핑이 **2 MiB PMD로 backing**된다.

```
FilePmdMapped:  2048 kB      <- 수정 전
weights RSS  :  2,097,152 / 4,194,304 B   (정확히 2 MiB 단위로 양자화)
```

그러면 128 KB든 512 KB든 부분 `MADV_DONTNEED`가 huge page를 쪼개지 못해 통째로
상주한 채 남는다.  이것 때문에 아래 시도가 **전부 무효**였다.

| 시도 | 결과 |
|---|---|
| prefetch 3 -> 2 block 재배정 | 변화 없음 |
| `MADV_SEQUENTIAL` -> `MADV_RANDOM` | 변화 없음 |
| block 512 KB -> 128 KB | 16 KB 개선 (무의미) |

`MADV_NOHUGEPAGE`를 걸어 `FilePmdMapped: 0 kB`를 확인한 뒤에야 block 크기와
prefetch 순서가 실제로 작동하기 시작했다.

## 수정한 코드

| 파일 | 변경 |
|---|---|
| `src/c/runtime/platform_linux/mapped_file.c` | `campp_mapped_file_advise_no_huge_page()` 추가 (`MADV_NOHUGEPAGE`) |
| `src/c/runtime/platform_linux/mapped_file.h` | 위 함수 선언 |
| `src/c/runtime/model_loading/weight_residency_loader.c:290` | windowing 진입 시 호출 |
| `src/python/.../planner/weight_streaming_planner.py` | 상한을 event timeline 기준으로 계산 / prefetch를 `block N-2` evict 이후로 |
| `scripts/5_model/08_build_weight_streaming_98.py` | `--bucket` 인자, 경로·파일명 파라미터화 |

### 새로 만든 도구

| 파일 | 용도 |
|---|---|
| `scripts/5_model/10_trace_weight_rss.py` | 외부에서 `/proc/<pid>/smaps`를 샘플링해 weights mapping RSS를 분리 추적.  peak 발생 **시점 특정용**이다 |
| `scripts/5_model/11_measure_weight_streaming_ram.py` | bucket별 V3 full vs windowed의 RAM/latency 측정 |

`10_trace_weight_rss.py`는 1~2 ms마다 smaps를 읽어 **측정을 교란한다**(VmHWM이
벤치마크보다 1 MB 정도 높게 나온다).  절대값을 벤치마크와 비교하지 말 것.

## 산출물 위치

```text
runs/models/campplus/final_v3/weight_streaming/{98,298,498,998}/
    plan_{bucket}.bin
    weights_{bucket}.bin
    weight_schedule_{bucket}.bin

results/models/campplus/final_v3/weight_streaming/
    ram_by_bucket.json          <- bucket별 최종 RAM/latency (핵심 결과)
    {bucket}/plan.json          <- block 배치와 상한
    98/benchmark_quick.json     <- 98 quick gate 결과
    98/rss_trace.json           <- peak 추적 타임라인
```

빌드 산출물: `build/profill/weight_streaming_98/` (98 전용으로 빌드했지만 다른
bucket에서도 동작한다 -- plan/schedule은 런타임 인자다).

## frame별 최종 RAM

### Peak RSS (VmHWM) -- OOM 기준, gate 지표

| frame | V3 full | windowed | 절감 | 절감율 |
|---|---|---|---|---|
| 98  | 11,935,744 | 6,418,432  | 5,517,312 | 46.2% |
| 298 | 15,335,424 | 9,850,880  | 5,484,544 | 35.8% |
| 498 | 18,714,624 | 13,193,216 | 5,521,408 | 29.5% |
| 998 | 27,205,632 | 21,786,624 | 5,419,008 | 19.9% |

### Current RSS -- 정상 상태

| frame | V3 full | windowed | 절감 | 절감율 |
|---|---|---|---|---|
| 98  | 11,935,744 | 4,390,912  | 7,544,832 | 63.2% |
| 298 | 15,335,424 | 7,794,688  | 7,540,736 | 49.2% |
| 498 | 18,714,624 | 11,198,464 | 7,516,160 | 40.2% |
| 998 | 27,205,632 | 19,673,088 | 7,532,544 | 27.7% |

### latency / RTF

| frame | full p50 | win p50 | Δ | full RTF | win RTF |
|---|---|---|---|---|---|
| 98  | 323.76  | 326.61  | +0.88% | 0.3242 | 0.3275 |
| 298 | 1123.16 | 1130.03 | +0.61% | 0.3741 | 0.3766 |
| 498 | 1825.58 | 1846.50 | +1.15% | 0.3654 | 0.3693 |
| 998 | 3709.95 | 3696.24 | -0.37% | 0.3704 | 0.3698 |

**절감 절대량이 bucket과 무관하게 5.42~5.52 MB로 일정하다.**  weight는 7.43 MB로
고정이고 bucket이 커질수록 activation arena만 늘기 때문이다.  긴 발화일수록 이
최적화의 상대 효과는 작아진다.

## block 크기 스윕 (huge page 제거 후, bucket 98)

| block | 개수 | bound | peak RSS | 절감 | RSS gate(>=5.5MB) |
|---|---|---|---|---|---|
| 512 KB | 16  | 1,048,576 | 7,049,216 | 4,874,240 | X |
| 128 KB | 66  | 638,976   | 6,565,888 | 5,378,048 | X |
| 64 KB  | 104 | 573,440   | 6,459,392 | 5,480,448 | X |
| **32 KB** | 146 | 552,960 | 6,434,816 | 5,509,120 | **O** |
| 16 KB  | 219 | 552,960   | 6,418,432 | 5,521,408 | O |
| 8 KB   | 328 | 552,960   | 6,451,200 | 5,484,544 | X |

8 KB에서 다시 나빠져 16~32 KB가 최적점이다.  **32 KB를 채택**했다 -- 16 KB와 RSS가
사실상 같은데 syscall이 3분의 2라 p50에 유리하다.  현재 4개 bucket 모두 32 KB다.

## 남은 작업

**Official 측정이 필요하다.**  98 quick gate는 RSS(>=5,500,000)를 6회 반복 전부
통과했지만 p50 Δ가 -0.17% ~ 1.69%로 흔들려 기준 1%를 넘나든다.  원인은 windowed가
아니라 **비교 기준인 `v3_full`이 322.3~325.7 ms로 변동**하는 데 있다 -- windowed는
326.9~328.1로 오히려 안정적이다.  quick(warmup 2 / repeat 10)의 표본 부족이다.

```bash
cd /home/arduino/workspace/egs/accelerate_CAM
python3 scripts/5_model/09_benchmark_weight_streaming_98.py --mode official --force
```

**bucket별 bitwise 검증이 아직 98만 되어 있다.**  `09_benchmark`가 retained tensor
832개 sha256과 embedding을 대조하는데, 298/498/998은 돌리지 않았다.  정식 승격 전에
각각 확인해야 한다.  `09_benchmark`도 98 하드코딩이 남아 있어 bucket 인자 추가가
필요하다.

**p50이 official에서도 1%를 넘으면** 다음 수단은 evict/prefetch 호출을 인접 block끼리
병합해 syscall 수를 줄이는 것이다.  현재 146 block에 대해 추론당 약 292회 호출한다.

## 환경

CPU governor를 `performance`(4코어 2016 MHz)로 고정해 두었다.  **재부팅하면
`schedutil`로 돌아간다.**  측정 재개 전 확인할 것.

```bash
for c in 0 1 2 3; do cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor; done
for c in 0 1 2 3; do echo performance | sudo tee \
  /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor; done
```

THP는 시스템 전역이 `[always]`다.  이 최적화는 `MADV_NOHUGEPAGE`로 해당 매핑만
예외 처리하므로 시스템 설정을 바꿀 필요는 없다.
