
import argparse, time
import numpy as np
from spp_experiments_v5 import SPPConfig, ModelConfig, train_rf, build_spp_context_features
from spp_experiments_v5_1 import select_feature_columns          
from spp_experiments_v5_timebased import build_spp_context_features_timebased
from credit_data_v8 import prepare_credit
from common_v8 import *

SPP = SPPConfig(enabled=True, levels=(1, 2), lookback_windows=20, context="past")


VARIANTS = [
    ("IAT/RateOnly + SPP",                 "iat_rate_only",    "count", "rf"),
    ("L7-Projected Baseline23 + SPP [27]", "baseline23",       "count", "rf"),
    ("Ours w/o Semantic (No Credit)",      "ours_no_semantic", "count", "rf"),
    ("Ours w/o Payload Div",               "ours_no_payload",  "count", "rf"),
    ("Full(Enhanced) + SPP (Ours)",        "ours_all",         "count", "rf"),
    ("L7-only 27D, no zero-pad",           "l7_only",          "count", "rf"),
    ("Full + time-based segmentation",     "ours_all",         "time",  "rf"),
    ("Full + XGBoost",                     "ours_all",         "count", "xgb"),
]
COLS = list(PATTERNS_ALL) + ["Macro", "WinFPR", "EntFPR", "EntFPR3", "AlertHr"]

def fit_predict(w, feat, seg, model, tr, va, te, seed):
    cols = select_feature_columns(w, feat)
    Xb = w[cols].to_numpy(float)
    if seg == "time":
        Xs = build_spp_context_features_timebased(w, cols, "api_key", "win_start", SPP, tr)
    else:
        Xs, _ = build_spp_context_features(w, cols, "api_key", "win_start", SPP, tr)
    X = np.hstack([Xb, Xs]); y = w["y_binary"].to_numpy(int)
    if model == "rf":
        clf = train_rf(X[tr], y[tr], ModelConfig(class_weight="balanced_subsample", random_state=seed))
    else:
        import xgboost as xgb
        pw = (len(y[tr]) - y[tr].sum()) / max(y[tr].sum(), 1)
        clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                scale_pos_weight=pw, random_state=seed, eval_metric="logloss")
        clf.fit(X[tr], y[tr])
    is_atk = w["is_atk"].to_numpy()
    p_va = clf.predict_proba(X[va])[:, 1]
    thr = calibrate_thr(p_va, (y[va] == 0) & (~is_atk[va]))
    p_te = clf.predict_proba(X[te])[:, 1]
    return (p_te > thr).astype(int), X.shape[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="traffic_logs.csv")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--holdout", default="")     
    ap.add_argument("--out", default="unified_v8.json")
    a = ap.parse_args()
    holdout = tuple(x for x in a.holdout.split(",") if x)

    df_raw, w, kind_map, atk, cfg, win_cfg = prepare_credit(a.csv)
    is_atk = w["is_atk"].to_numpy()
    print(f"[*] windows={len(w)} attack_windows={int(w.y_binary.sum())} "
          f"entities={w.api_key.nunique()} attackers={len(atk)}")
    print(w.loc[w.y_binary == 1, "type_mode"].value_counts().to_string())

    res = {v[0]: [] for v in VARIANTS}; dims = {}
    for seed in SEEDS[:a.seeds]:
        t0 = time.time()
        trk, vak, tek = stratified_entity_split(kind_map, seed, holdout)
        idx = lambda S: w.index[w["api_key"].isin(S)].to_numpy()
        tr, va, te = idx(trk), idx(vak), idx(tek)
        for name, feat, seg, model in VARIANTS:
            try:
                pred, dim = fit_predict(w, feat, seg, model, tr, va, te, seed)
            except ImportError:
                print(f"    [skip] {name}: xgboost 未安装"); continue
            dims[name] = dim
            res[name].append(compute_metrics(w.iloc[te], pred, is_atk[te]))
        n_te_atk = len([k for k in tek if k in atk])
        print(f"[seed={seed}] test entities: {len(tek)} (attackers {n_te_atk}) | {time.time()-t0:.0f}s")

    summary = print_table(res, COLS)
    print("\n融合后维度:", {k: v for k, v in dims.items()})
    save_json(a.out, {"summary": summary, "per_seed": res, "fused_dim": dims,
                      "protocol": "hashed keys, stratified entity-disjoint 70/10/20, thr@1% normal-FPR on val"})

if __name__ == "__main__":
    main()