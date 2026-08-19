# QConv v5 address module

이 디렉터리는 MAC multiply 및 dispatch 구현과 분리된 address producer다.

```text
address/
├── qconv_v5_address_contract.h   # address producer와 MAC consumer의 중립 POD
├── qconv_v5_address.h            # plan, row schedule, tile provider API
├── qconv_v5_address_internal.h   # checked offset helper
├── qconv_v5_address_plan.c       # geometry 검증과 파생 step/interior 계산
├── qconv_v5_address_schedule.c   # `/`, `%` 없는 row-wise work item
├── qconv_v5_address_1x1.c        # base + output-step pointer walk
├── qconv_v5_address_3x3.c        # base + kernel/output-step pointer walk
└── qconv_v5_address_generic.c    # border, padding, tail의 명시적 NULL
```

의존 방향은 다음과 같다.

```text
planning ──> address API ──> address contract
dispatch ──> planning + address API
MAC ───────────────────────> address contract
```

Address 구현은 V4/V5 execution plan, parameter, weight pack, requant, store 또는
MAC header를 include하지 않는다. `planning/qconv_v5_execution_plan.*`만 runtime
Tensor view를 중립 geometry로 변환한다.

현재 구현 범위:

- op 단위 output/kernel byte step 및 interior 범위 사전 계산
- 전체 tile `memset` 제거
- 1x1 증분 pointer와 최초/최종 storage 검사
- 3x3 interior의 point별 72회 검사 대신 최초/최종 2회 검사
- row cursor를 통한 left/interior/right 분리와 행 경계 보장
- generic padding point의 명시적 `NULL`

Direct-stride와 sliding-window는 MAC ABI 변경이 필요하므로 기존 pointer-table
contract와 별도 gate로 연결한다.
