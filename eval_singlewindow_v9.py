# eval_singlewindow_v9.py
# 单窗口 RF vs RF+SPP 消融（回应 R2 对 SPP 作用的质疑）
# 不修改任何旧文件；协议与 eval_unified_v8.py 完全一致：
#   哈希 key、分层实体切分 70/10/20、验证集 1% 正常窗口 FPR 校准、种子 42-51
# 用法: python eval_singlewindow_v9.py --seeds 1    (冒烟)
#       python eval_singlewindow_v9.py --seeds 10   (正式)
import argparse, time
import numpy as np
from spp_experiments_v5 import SPPConfig, ModelConfig, train_rf, build_spp_context_features
from spp_experiments_v5_1 import select_feature_columns   # 含 l7_only，其它透传
from credit_data_v8 import prepare_credit
from common_v8 import *

SPP = SPPConfig(enabled=True, levels=(1, 2), lookback_windows=20, context="past")

#        显示名                                特征集       是否拼接SPP
VARIANTS = [
    ("Single-window RF, 50D",              "ours_all", False),
    ("Single-window RF, 27D (L7-only)",    "l7_only",  False),
    ("RF + SPP, 50D (reference = v8)",     "ours_all", True),
    ("RF + SPP, 27D (L7-only)",            "l7_only",  True),
]
COLS = list(PATTERNS_ALL) + ["Macro", "WinFPR", "EntFPR", "EntFPR3", "AlertHr"]

def fit_predict(w, feat, use_spp, tr, va, te, seed):
    cols = select_feature_columns(w, feat)
    Xb = w[cols].to_numpy(float)
    if use_spp:
        Xs, _ = build_spp_context_features(w, cols, "api_key", "win_start", SPP, tr)
        X = np.hstack([Xb, Xs])
    else:
        X = Xb                                   # 唯一区别：不拼接 SPP 特征
    y = w["y_binary"].to_numpy(int); is_atk = w["is_atk"].to_numpy()
    clf = train_rf(X[tr], y[tr], ModelConfig(class_weight="balanced_subsample", random_state=seed))
    thr = calibrate_thr(clf.predict_proba(X[va])[:, 1], (y[va] == 0) & (~is_atk[va]))
    pred = (clf.predict_proba(X[te])[:, 1] > thr).astype(int)
    return pred, X.shape[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="traffic_logs.csv")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="singlewindow_v9.json")
    a = ap.parse_args()

    df_raw, w, kind_map, atk, cfg, win_cfg = prepare_credit(a.csv)
    is_atk = w["is_atk"].to_numpy()
    print(f"[*] windows={len(w)} attack_windows={int(w.y_binary.sum())} "
          f"entities={w.api_key.nunique()} attackers={len(atk)}")

    res = {v[0]: [] for v in VARIANTS}; dims = {}
    for seed in SEEDS[:a.seeds]:
        t0 = time.time()
        trk, vak, tek = stratified_entity_split(kind_map, seed)
        idx = lambda S: w.index[w["api_key"].isin(S)].to_numpy()
        tr, va, te = idx(trk), idx(vak), idx(tek)
        for name, feat, use_spp in VARIANTS:
            pred, dim = fit_predict(w, feat, use_spp, tr, va, te, seed)
            dims[name] = dim
            res[name].append(compute_metrics(w.iloc[te], pred, is_atk[te]))
        print(f"[seed={seed}] done | {time.time()-t0:.0f}s")

    summary = print_table(res, COLS, name_w=38)
    print("\n输入维度:", dims)
    save_json(a.out, {"summary": summary, "per_seed": res, "input_dim": dims,
                      "protocol": "same as unified_v8: hashed keys, stratified entity-disjoint 70/10/20, thr@1% normal-FPR on val"})

if __name__ == "__main__":
    main()