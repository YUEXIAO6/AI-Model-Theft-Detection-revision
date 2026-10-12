# cold_start_v13.py —— 窗口在 key 历史中的序号 vs 误报 / 首次告警位置
import argparse, numpy as np, pandas as pd
from common_v8 import *
from credit_data_v8 import prepare_credit
from eval_unified_v8 import fit_predict

BUCKETS = [(0, 2), (3, 4), (5, 9), (10, 19), (20, 10**9)]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); a = ap.parse_args()
    _, w, kind_map, atk, _, _ = prepare_credit()
    w = w.sort_values(["api_key", "win_start"]).reset_index(drop=True) if False else w
    # 序号 = 该 key 的第几个活跃窗口（按时间）
    w["hist_idx"] = w.groupby("api_key")["win_start"].rank(method="first").astype(int) - 1
    is_atk = w["is_atk"].to_numpy(); y = w["y_binary"].to_numpy()
    rows, first_pos = [], {"benign": [], "attacker": []}
    for seed in SEEDS[:a.seeds]:
        trk, vak, tek = stratified_entity_split(kind_map, seed)
        idx = lambda S: w.index[w["api_key"].isin(S)].to_numpy()
        tr, va, te = idx(trk), idx(vak), idx(tek)
        pred, _ = fit_predict(w, "ours_all", "count", "rf", tr, va, te, seed)
        d = w.iloc[te].copy(); d["pred"] = pred
        neg = d[(d.y_binary == 0) & (~d.is_atk)]
        for lo, hi in BUCKETS:
            m = neg[(neg.hist_idx >= lo) & (neg.hist_idx <= hi)]
            rows.append({"seed": seed, "bucket": f"{lo}-{hi if hi < 10**8 else '+'}",
                         "n": len(m), "fpr": m.pred.mean() if len(m) else np.nan})
        for k, g in d.groupby("api_key"):
            f = g[g.pred == 1]
            if len(f):
                first_pos["attacker" if g.is_atk.iloc[0] else "benign"].append(int(f.hist_idx.min()))
    r = pd.DataFrame(rows).groupby("bucket", sort=False).agg(n=("n", "sum"), fpr=("fpr", "mean"))
    print("\n良性正常窗口 FPR 按历史序号分桶:\n", (r.assign(fpr=r.fpr * 100)).round(2).to_string())
    for k, v in first_pos.items():
        v = np.array(v)
        if len(v):
            print(f"{k}: 首次告警的历史序号 中位数={np.median(v):.0f}, 均值={v.mean():.1f}, "
                  f"序号<5 的占比={np.mean(v < 5)*100:.0f}%  (n={len(v)})")
    save_json("cold_start_v13.json", {"fpr_by_bucket": r.reset_index().to_dict("records"),
                                      "first_flag_idx": first_pos})

if __name__ == "__main__":
    main()