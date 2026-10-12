# eval_tenant_v13.py
import argparse
from common_v8 import *
from credit_data_v8 import prepare_credit
from eval_unified_v8 import fit_predict

COLS = ["S0", "S1", "S2", "S3", "S4", "S6", "Macro", "WinFPR", "EntFPR3"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--hold", default="atk_s4,atk_s6")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="tenant_v13.json")
    a = ap.parse_args()
    HOLD = tuple(x for x in a.hold.split(",") if x)
    res = {}
    for csv in a.csvs:
        df_raw, w, kind_map, atk, _, _ = prepare_credit(csv)
        is_atk = w["is_atk"].to_numpy()
        for seed in SEEDS[:a.seeds]:
            trk, vak, tek = stratified_entity_split(kind_map, seed, HOLD)
            if seed == SEEDS[0]:
                cnt = {}
                for k in tek:
                    cnt[kind_map[k]] = cnt.get(kind_map[k], 0) + 1
                print(f"[{csv}] test entities by kind (seed {seed}):",
                      {k: v for k, v in sorted(cnt.items()) if k.startswith("atk")})
            idx = lambda S: w.index[w["api_key"].isin(S)].to_numpy()
            tr, va, te = idx(trk), idx(vak), idx(tek)
            pred, _ = fit_predict(w, "ours_all", "count", "rf", tr, va, te, seed)
            res.setdefault(f"{csv} | hold={a.hold}", []).append(
                compute_metrics(w.iloc[te], pred, is_atk[te]))
            print(f"  [seed={seed}] done")
    summary = print_table(res, COLS, name_w=70)
    save_json(a.out, {"summary": summary, "per_seed": res, "holdout": HOLD})

if __name__ == "__main__":
    main()