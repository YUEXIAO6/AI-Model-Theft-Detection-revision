# server_insurance/eval_insurance_v8.py
# [v8-fix] COLS 增加 EntFPR3
import sys, os, argparse; sys.path.insert(0, os.path.abspath(".."))   # 复用上级目录的 common_v8
import numpy as np
from spp_experiments_insurance import (ExperimentConfig, WindowConfig, SPPConfig, ModelConfig,
    load_and_prepare_df, build_windows_df, train_rf, select_feature_columns, build_spp_context_features)
from window_relabel_credit_only import relabel_windows_credit_only
from common_v8 import *

SPP = SPPConfig(enabled=True, levels=(1, 2), lookback_windows=20, context="past")
VARIANTS = [("IAT/RateOnly + SPP", "iat_rate_only"), ("L7-Projected Baseline23 + SPP", "baseline23"),
            ("Ours w/o Payload Div", "ours_no_payload"), ("Full(Enhanced) + SPP", "ours_all")]
COLS = list(PATTERNS_ALL) + ["Macro", "WinFPR", "EntFPR", "EntFPR3", "AlertHr"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10); a = ap.parse_args()
    cfg = ExperimentConfig(csv_file="traffic_logs_insurance.csv", group_col="api_key")
    win_cfg = WindowConfig(window_s=60, attack_ratio_thr=0.1)
    df_raw = load_and_prepare_df(cfg); df_raw, kind_map, atk = hash_keys(df_raw)
    w = build_windows_df(df_raw, cfg, win_cfg, True, True)
    w = relabel_windows_credit_only(df_raw=df_raw, windows_df=w, window_s=60, group_col="api_key",
                                    credit_endpoint="/api/v1/insurance_pricing").sort_values("win_start").reset_index(drop=True)
    w["is_atk"] = w["api_key"].isin(atk); y = w["y_binary"].to_numpy(int); is_atk = w["is_atk"].to_numpy()
    print(f"[*] windows={len(w)} attack_windows={int(w.y_binary.sum())} entities={w.api_key.nunique()} attackers={len(atk)}")
    print(w.loc[w.y_binary == 1, "type_mode"].value_counts().to_string())
    res = {v[0]: [] for v in VARIANTS}
    for seed in SEEDS[:a.seeds]:
        trk, vak, tek = stratified_entity_split(kind_map, seed)
        idx = lambda S: w.index[w["api_key"].isin(S)].to_numpy(); tr, va, te = idx(trk), idx(vak), idx(tek)
        for name, feat in VARIANTS:
            cols = select_feature_columns(w, feat); Xb = w[cols].to_numpy(float)
            Xs, _ = build_spp_context_features(w, cols, "api_key", "win_start", SPP, tr); X = np.hstack([Xb, Xs])
            clf = train_rf(X[tr], y[tr], ModelConfig(class_weight="balanced_subsample", random_state=seed))
            thr = calibrate_thr(clf.predict_proba(X[va])[:, 1], (y[va] == 0) & (~is_atk[va]))
            res[name].append(compute_metrics(w.iloc[te], (clf.predict_proba(X[te])[:, 1] > thr).astype(int), is_atk[te]))
        n_te_atk = len([k for k in tek if k in atk])
        print(f"[seed={seed}] test entities: {len(tek)} (attackers {n_te_atk})")
    summary = print_table(res, COLS)
    save_json("insurance_v8.json", {"summary": summary, "per_seed": res})   # 存到 server_insurance/results_v8/

if __name__ == "__main__":
    main()