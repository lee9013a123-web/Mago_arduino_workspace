# E7 Operator Profiling

E7 Bucket 98의 `kernel->run()` exclusive wall time을 Operator ID별로
측정하고, end-to-end latency의 누적 80%를 차지하는 최소 Operator 집합을 만든다.

C 계측기는 시간과 최소 식별자만 출력한다. opcode, dtype, Tensor·weight shape와
fusion 원본 정보는 기존 `read_execution_plan()`과
`fusion_plans/fusion_98.json`을 Python에서 조인한다.

## 측정 범위

```text
Tensor view 준비       제외
bounds check           제외
clock start
kernel->run()          포함
clock end
bounds check/callback  제외
```

end-to-end 시간은 `campp_graph_execute()` 전체를 별도로 감싼다. 따라서
`end-to-end - kernel exclusive 합`은 executor와 bounds check 등을 포함하는
`unattributed_runtime`으로 남는다.

## 빌드

Linux 또는 QRB2210 보드에서 실행한다.

```bash
bash scripts/4_profill/01_build_profiler.sh
build/profill/test_operator_profiler
build/profill/test_profiled_graph_executor
python3 -m unittest tests.runtime.profill.test_profile_statistics
```

기본 빌드 옵션은 두 실행 파일 모두
`-std=c11 -O3 -DNDEBUG -Wall -Wextra`이다.

## 사전 검사

```bash
python3 scripts/4_profill/02_profile_e7.py --preflight-only
```

다음을 강제한다.

- Bucket 98
- CPU affinity `[0]`
- 실제 thread 1개
- warm-up 20회
- 측정 100회
- 입력 `multi__speaker_0000`, `0005`, `0006`
- 기존 E7 bitwise validation 통과

## 정식 실행

```bash
python3 scripts/4_profill/02_profile_e7.py
```

기존 결과를 의도적으로 교체할 때만 `--force`를 사용한다.

Raw 결과:

```text
runs/profiling/e7_98/raw/
```

최종 결과:

```text
results/profiling/e7_98/operator_profile.json
results/profiling/e7_98/operator_profile.csv
results/profiling/e7_98/summary.json
```

`Top bottleneck set`은 Operator exclusive time 내림차순의 최소 prefix이며,
누적 end-to-end 비중이 80%에 도달해야 `complete=true`가 된다. 모든 kernel
exclusive time을 더해도 80% 미만이면 이를 100%로 재정규화하지 않고
`complete=false`와 `unattributed_runtime`을 보고한다.

계측 overhead는 같은 Release 옵션의 비계측 binary와 비교한다. 1% 이상이면
결과 파일은 생성하지만 프로세스는 종료 코드 3을 반환하고
`profile_valid=false`로 기록한다.
