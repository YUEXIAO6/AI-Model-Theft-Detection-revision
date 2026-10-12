# eval_noserver_v12.py
import argparse
from common_v8 import *
from credit_data_v8 import prepare_credit
import eval_unified_v8 as eu
from spp_experiments_v5_1 import select_feature_columns as _base

SERVER = ["proc_dur_mean", "proc_dur_std", "status_2xx_frac", "status_4xx_frac", "status_5xx_frac"]
def sel(w, fs):
    if fs == "ours_no_server":
        return [c for c in _base(w, "ours_all") if c not in SERVER]
    return _base(w, fs)
eu.select_feature_columns = sel        

COLS = ["S0", "S1", "S2", "S3", "Macro", "WinFPR", "EntFPR3", "AlertHr"]
VARIANTS = [("Full 50D", "ours_all", 0.0),
            ("No server-side features (45D)", "ours_no_server", 0.0),
            ("Full 50D, test latency +0.11 s", "ours_all", 0.11)]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10); a = ap.parse_args()
    df_raw, w, kind_map, atk, _, _ = prepare_credit()
    is_atk = w["is_atk"].to_numpy(); res = {v[0]: [] for v in VARIANTS}
    for seed in SEEDS[:a.seeds]:
        trk, vak, tek = stratified_entity_split(kind_map, seed)
        idx = lambda S: w.index[w["api_key"].isin(S)].to_numpy()
        tr, va, te = idx(trk), idx(vak), idx(tek)
        for name, feat, shift in VARIANTS:
            wv = w.copy()
            if shift: wv.loc[te, "proc_dur_mean"] += shift     
            pred, dim = eu.fit_predict(wv, feat, "count", "rf", tr, va, te, seed)
            res[name].append(compute_metrics(w.iloc[te], pred, is_atk[te]))
        print(f"[seed={seed}] done")
    summary = print_table(res, COLS, name_w=36)
    save_json("noserver_v12.json", {"summary": summary, "per_seed": res})

if __name__ == "__main__":
    main()