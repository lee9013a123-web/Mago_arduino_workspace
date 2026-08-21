"""98 operator profile을 weight tensor id 기준으로 다른 bucket에 재매핑한다.

winner를 옮기는 것이 아니라 '무엇을 측정할지'만 옮긴다.  latency는 각 bucket에서
실제로 다시 측정한다.  operator_id는 bucket마다 다르지만 weight tid는 상수이므로
이 매핑으로 대상 op를 정확히 지목할 수 있다.
"""
import copy, json, pathlib

ROOT = pathlib.Path("/home/arduino/workspace/egs/accelerate_CAM")
FP = ROOT / "runs/runtime/kernel_optimization/e7/bundle/fusion_plans"
SRC = ROOT / "results/profiling/e7_98/operator_profile.json"
BUCKETS = [298, 498, 998]
WEIGHT_INPUT = 3


def exec_map(bucket):
    f = json.load(open(FP / f"fusion_{bucket}.json"))
    by_op, by_wt = {}, {}
    for e in f["execution_table"]:
        if e["opcode"] != "QLINEAR_CONV":
            continue
        wt = e["input_tensor_ids"][WEIGHT_INPUT]
        by_op[e["operator_id"]] = (wt, e["kernel_id"])
        by_wt[(e["kernel_id"], wt)] = e["operator_id"]
    return by_op, by_wt


prof = json.load(open(SRC))
base_op, _ = exec_map(98)
ops = prof["operators"]
conv_names = {"qlinear_conv_o4i4_neon", "fused_quant_qlinear_conv_o4i4"}

for b in BUCKETS:
    _, wt_to_op = exec_map(b)
    out = copy.deepcopy(prof)
    kept, dropped = [], 0
    for o in out["operators"]:
        oid = int(o["operator_id"])
        if o.get("kernel_name") not in conv_names:
            continue                       # conv 외 연산은 이 profile에서 쓰지 않음
        if oid not in base_op:
            dropped += 1
            continue
        wt, kid = base_op[oid]
        new_oid = wt_to_op.get((kid, wt))
        if new_oid is None:
            dropped += 1
            continue
        o["operator_id"] = new_oid
        kept.append(o)
    kept.sort(key=lambda x: int(x["operator_id"]))
    out["operators"] = kept
    out["bucket_frames"] = b
    out["derived_from"] = {
        "source_profile": "results/profiling/e7_98/operator_profile.json",
        "method": "weight tensor id remap (operator_id만 재매핑, latency는 재측정 대상)",
        "note": "mean_ms/share_pct는 98 값이며 시간 추정용일 뿐 선택 근거가 아니다",
    }
    d = ROOT / f"results/profiling/e7_{b}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "operator_profile.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ordinary = sum(1 for o in kept if o["kernel_name"] == "qlinear_conv_o4i4_neon")
    fused = len(kept) - ordinary
    print(f"bucket {b:>4}: ordinary {ordinary}  fused {fused}  누락 {dropped}"
          f"  -> {d.relative_to(ROOT)}/operator_profile.json")

    # 재매핑된 op가 대상 plan에서 실제로 유효한지 검증
    chk_op, _ = exec_map(b)
    bad = [o["operator_id"] for o in kept
           if o["kernel_name"] == "qlinear_conv_o4i4_neon"
           and int(o["operator_id"]) not in chk_op]
    print(f"           plan 유효성: 무효 op {len(bad)}개")
