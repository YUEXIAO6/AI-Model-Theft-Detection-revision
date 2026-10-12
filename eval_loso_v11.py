
import argparse, time
import numpy as np
import common_v11
from common_v8 import *
from credit_data_v8 import prepare_credit
from eval_unified_v8 import fit_predict as fit_spp
from eval_singlewindow_v9 import fit_predict as fit_single

CONFIGS = [("none", ()), ("hold S0", ("atk_s0",)), ("hold S1", ("atk_s1",)),
           ("hold S2", ("atk_s2",)), ("hold S1+S2", ("atk_s1", "atk_s2")), ("hold S3", ("atk_s3",))]
VARIANTS = [("Full50D+SPP", "ours_all", "spp"), ("L7only27D+SPP", "l7_only", "spp"),
            ("Full50D single-window", "ours_all", "single")]
COLS = ["S0", "S1", "S2", "S3", "Macro", "WinFPR", "EntFPR3", "AlertHr"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="traffic_logs.csv")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--variants", default="0,1,2")     
    ap.add_argument("--out", default="loso_v11.json")
    a = ap.parse_args()
    vsel = [VARIANTS[int(i)] for i in a.variants.split(",")]

    df_raw, w, kind_map, atk, cfg, win_cfg = prepare_credit(a.csv)
    is_atk = w["is_atk"].to_numpy()
    res = {}
    for seed in SEEDS[:a.seeds]:
        t0 = time.time()
        for cname, hold in CONFIGS:
            trk, vak, tek = stratified_entity_split(kind_map, seed, hold)
            idx = lambda S: w.index[w["api_key"].isin(S)].to_numpy()
            tr, va, te = idx(trk), idx(vak), idx(tek)
            for vname, feat, mode in vsel:
                if mode == "spp": pred, _ = fit_spp(w, feat, "count", "rf", tr, va, te, seed)
                else:             pred, _ = fit_single(w, feat, False, tr, va, te, seed)
                res.setdefault(f"{cname:<11}| {vname}", []).append(compute_metrics(w.iloc[te], pred, is_atk[te]))
        print(f"[seed={seed}] done | {time.time()-t0:.0f}s")

    summary = print_table(res, COLS, name_w=36)
    
    save_json(a.out, {"summary": summary, "per_seed": res})

if __name__ == "__main__":
    main()