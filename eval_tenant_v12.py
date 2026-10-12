# eval_tenant_v12.py —— S4/S6 作为 zero-day 留出，其余协议与 eval_unified_v8 完全一致
import argparse
from common_v8 import *
from credit_data_v8 import prepare_credit
from eval_unified_v8 import fit_predict

COLS = ["S0", "S1", "S2", "S3", "S4", "S6", "Macro", "WinFPR", "EntFPR3"]
HOLD = ("atk_s4", "atk_s6")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    res = {}
    for csv in a.csvs:
        df_raw, w, kind_map, atk, _, _ = prepare_credit(csv)
        is_atk = w["is_atk"].to_numpy()
        kc = {}
        for k, kd in kind_map.items(): kc[kd] = kc.get(kd, 0) + 1
        print(f"\n[*] {csv}: windows={len(w)} entities={kc}")
        for seed in SEEDS[:a.seeds]:
            trk, vak, tek = stratified_entity_split(kind_map, seed, HOLD)
            idx = lambda S: w.index[w["api_key"].isin(S)].to_numpy()
            tr, va, te = idx(trk), idx(vak), idx(tek)
            pred, _ = fit_predict(w, "ours_all", "count", "rf", tr, va, te, seed)
            res.setdefault(csv, []).append(compute_metrics(w.iloc[te], pred, is_atk[te]))
            print(f"  [seed={seed}] done")
    summary = print_table(res, COLS, name_w=40)
    save_json("tenant_v12.json", {"summary": summary, "per_seed": res, "holdout": HOLD})

if __name__ == "__main__":
    main()