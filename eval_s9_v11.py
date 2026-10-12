
import argparse, time
import numpy as np, pandas as pd
import common_v11                                  
from common_v8 import *
from credit_data_v8 import prepare_credit
from eval_unified_v8 import fit_predict as fit_spp
from eval_singlewindow_v9 import fit_predict as fit_single

VARIANTS = [
    ("L7-projected Baseline23 + SPP", "baseline23", "spp"),
    ("Full 50D + SPP",                "ours_all",   "spp"),
    ("L7-only 27D + SPP",             "l7_only",    "spp"),
    ("Full 50D, single window",       "ours_all",   "single"),
]
COLS = ["S0", "S1", "S2", "S3", "S5", "S9", "Macro03", "WinFPR", "EntFPR3", "AlertHr"]
CRED = ["credit_unique_age", "credit_unique_income", "credit_unique_pairs", "credit_age_entropy",
        "credit_income_entropy", "credit_age_step_mean", "credit_income_step_mean"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="traffic_logs_v10b.csv")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--holdout", default="atk_s5,atk_s9")   
    ap.add_argument("--out", default="s9_v11.json")
    a = ap.parse_args()
    holdout = () if a.holdout in ("", "none") else tuple(x for x in a.holdout.split(",") if x)

    df_raw, w, kind_map, atk, cfg, win_cfg = prepare_credit(a.csv)
    is_atk = w["is_atk"].to_numpy()
    print(f"[*] windows={len(w)} attack_windows={int(w.y_binary.sum())} entities={w.api_key.nunique()} "
          f"attackers={len(atk)} holdout={holdout}")
    print(w.loc[w.y_binary == 1, "type_mode"].value_counts().to_string())
    d = w[w["credit_unique_age"] > 0]
  
    print(d.groupby("type_mode")[CRED].mean().round(2).to_string())

    res = {v[0]: [] for v in VARIANTS}
    for seed in SEEDS[:a.seeds]:
        t0 = time.time()
        trk, vak, tek = stratified_entity_split(kind_map, seed, holdout)
        idx = lambda S: w.index[w["api_key"].isin(S)].to_numpy()
        tr, va, te = idx(trk), idx(vak), idx(tek)
        for name, feat, mode in VARIANTS:
            if mode == "spp": pred, _ = fit_spp(w, feat, "count", "rf", tr, va, te, seed)
            else:             pred, _ = fit_single(w, feat, False, tr, va, te, seed)
            m = compute_metrics(w.iloc[te], pred, is_atk[te])
            v = [m[k] for k in ("S0", "S1", "S2", "S3") if not np.isnan(m[k])]
            m["Macro03"] = float(np.mean(v)) if v else np.nan
            res[name].append(m)
        print(f"[seed={seed}] test attackers={len([k for k in tek if k in atk])} | {time.time()-t0:.0f}s")

    summary = print_table(res, COLS, name_w=32)
    save_json(a.out, {"summary": summary, "per_seed": res, "holdout": holdout})

if __name__ == "__main__":
    main()