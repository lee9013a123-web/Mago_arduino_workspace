"""측정이 완료된 bucket만으로 conv_layer_hybrid_plan.c를 생성한다.

측정하지 않은 bucket은 테이블을 넣지 않는다 -- 그래야 dispatch가 V2(v4 /
combined_hybrid)로 안전하게 fallback한다.  전이(가설) 테이블을 소스에 남기면
나중에 측정된 것으로 오해할 수 있으므로 넣지 않는다.
"""
import json, pathlib, sys

ROOT = pathlib.Path("/home/arduino/workspace/egs/accelerate_CAM")
SRC = ROOT / ("src/c/profill/optimization/candidates/conv_layer_hybrid/"
              "conv_layer_hybrid_plan.c")
MEASURED = [int(b) for b in sys.argv[1:]] or [98]
MARGIN = 1.0
INCUMBENT = {"qconv": "v4", "fused": "combined_hybrid"}
KEYMAP = {"mac_fixed": "qconv_mac_fixed", "v5": "qconv_v5", "v4": None,
          "combined_fixed": "fused_fixed", "combined_v5": "fused_v5",
          "combined_hybrid": None}
NAMES = [("qconv_mac_fixed", "CAMPP_QCONV_MAC_FIXED_IDS"),
         ("qconv_v5", "CAMPP_QCONV_V5_IDS"),
         ("fused_fixed", "CAMPP_FUSED_QCONV_FIXED_IDS"),
         ("fused_v5", "CAMPP_FUSED_QCONV_V5_IDS")]


def winners(results_dir, modes, incumbent):
    """comparison JSON에서 op별 1% margin winner를 뽑는다."""
    lat = {}
    for m in modes:
        p = results_dir / f"{m}_comparison.json"
        for c in json.load(open(p))["cases"]:
            lat.setdefault(c["operator_id"], {})[m] = c["candidate_mean_ms"]
    out = {}
    for oid, mm in lat.items():
        if incumbent not in mm:
            continue
        best = min(mm, key=mm.get)
        gain = (mm[incumbent] - mm[best]) / mm[incumbent] * 100.0
        out[oid] = best if gain >= MARGIN else incumbent
    return out


tables, report = {}, []
for b in MEASURED:
    base = ROOT / f"results/profiling/e7_{b}/optimization"
    t = {k: [] for k, _ in NAMES}
    q = winners(base / "qconv_family", ("mac_fixed", "v4", "v5"), "v4")
    f = winners(base / "fused_qconv_family",
                ("combined_fixed", "combined_hybrid", "combined_v5"),
                "combined_hybrid")
    for oid, w in sorted(q.items()):
        k = KEYMAP[w]
        if k:
            t[k].append(oid)
    for oid, w in sorted(f.items()):
        k = KEYMAP[w]
        if k:
            t[k].append(oid)
    tables[b] = t
    report.append((b, len(q), len(f), {k: len(v) for k, v in t.items()}))

def arr(name, ids):
    ids = sorted(ids)
    body = "".join("    " + " ".join(f"{v}u," for v in ids[i:i + 10]) + "\n"
                   for i in range(0, len(ids), 10))
    return f"static const uint32_t {name}[] = {{\n{body}}};\n\n"

out = ['#include "conv_layer_hybrid_plan.h"\n\n#include <stddef.h>\n'
       '#include <stdint.h>\n\n',
       "/* Operator 단위 latency 실측으로 고른 layer-hybrid plan.\n"
       " * 각 bucket에서 mac_fixed/v4/v5(그리고 fused 3종)를 직접 재고 incumbent보다\n"
       " * 1%% 이상 빠른 layer만 교체한다.  측정하지 않은 bucket은 테이블이 없어\n"
       " * dispatch가 v4 / combined_hybrid로 fallback한다.\n"
       f" * 측정 완료 bucket: {', '.join(str(b) for b in MEASURED)} */\n\n"]
for b in MEASURED:
    for key, base in NAMES:
        out.append(arr(f"{base}_{b}", tables[b][key]))
out.append("""typedef struct {
    uint32_t bucket_frames;
    const uint32_t *qconv_mac_fixed_ids;
    size_t qconv_mac_fixed_count;
    const uint32_t *qconv_v5_ids;
    size_t qconv_v5_count;
    const uint32_t *fused_fixed_ids;
    size_t fused_fixed_count;
    const uint32_t *fused_v5_ids;
    size_t fused_v5_count;
} CamppConvLayerHybridBucketPlan;

static const CamppConvLayerHybridBucketPlan CAMPP_BUCKET_PLANS[] = {
""")
for b in MEASURED:
    e = [f"        {b}u,"]
    for key, base in NAMES:
        n = f"{base}_{b}"
        e.append(f"        {n}, sizeof({n}) / sizeof({n}[0]),")
    out.append("    {\n" + "\n".join(e) + "\n    },\n")
out.append("};\n")

old = SRC.read_text()
SRC.write_text("".join(out) + "\n"
               + old[old.index("static int campp_id_in_sorted_table("):])
for b, nq, nf, cnt in report:
    print(f"bucket {b:>4}: qconv {nq}개 / fused {nf}개 측정, 테이블 {cnt}")
print(f"생성 완료 (bucket {MEASURED})")
