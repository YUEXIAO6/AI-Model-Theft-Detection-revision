# mit_eval_v10_delay.py —— 响应策略：零延迟 / 1 窗口 / 2 窗口因果延迟
import argparse, zlib
import numpy as np, pandas as pd
import common_v10                                   # 必须第一个导入
from common_v8 import *
from spp_experiments_v5 import ModelConfig, train_rf, select_feature_columns, build_spp_context_features
from credit_data_v8 import prepare_credit
from mit_state_machine_v2 import simulate_entity_policy_v2
from mit_eval_v8 import CONFIGS, SPP, target_model, surrogate_mse, admitted, first_action_queries

def shift_policy(pol, d):
    """窗口 i 的实际状态 = 窗口 i-d 的决策；前 d 个窗口放行"""
    if d == 0: return pol
    return [(t, "normal", 1.0) if i < d else (t, pol[i - d][1], pol[i - d][2])
            for i, (t, st, ad) in enumerate(pol)]

def cell(s, kind):
    m, sd, n = s
    if n == 0: return "n/a"
    if kind == "pct": return f"{m*100:5.1f}±{sd*100:4.1f}"
    if kind == "mse": return f"{m:.4f}±{sd:.4f}"
    return f"{m:6.2f}±{sd:5.2f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--delays", default="0,1,2")
    a = ap.parse_args()
    DELAYS = [int(x) for x in a.delays.split(",")]

    df_raw, w, kind_map, atk, _, _ = prepare_credit()
    y = w["y_binary"].to_numpy(int); is_atk = w["is_atk"].to_numpy()
    cols = select_feature_columns(w, "ours_all"); Xb = w[cols].to_numpy(float)
    q_all = df_raw[df_raw["endpoint"] == "/api/v1/credit_score"].copy()
    q_all["age"] = pd.to_numeric(q_all["age"], errors="coerce")
    q_all["income"] = pd.to_numeric(q_all["income"], errors="coerce")
    q_all = q_all.dropna(subset=["age", "income"])
    q_all["timestamp"] = to_epoch_s(q_all["timestamp"]); q_all = q_all.dropna(subset=["timestamp"])

    rng0 = np.random.RandomState(123)
    X_test = np.column_stack((rng0.uniform(18, 90, 1000), rng0.uniform(10000, 120000, 1000)))
    y_test = target_model(X_test[:, 0], X_test[:, 1])
    base_mse = float(np.mean((y_test - 0.5) ** 2))
    print(f"[*] no-knowledge baseline MSE = {base_mse:.4f}")

    rows = {(c, d): [] for c in CONFIGS for d in DELAYS}
    for seed in SEEDS[:a.seeds]:
        trk, vak, tek = stratified_entity_split(kind_map, seed)
        idx = lambda S: w.index[w["api_key"].isin(S)].to_numpy()
        tr, va, te = idx(trk), idx(vak), idx(tek)
        Xs, _ = build_spp_context_features(w, cols, "api_key", "win_start", SPP, tr)
        X = np.hstack([Xb, Xs])
        clf = train_rf(X[tr], y[tr], ModelConfig(class_weight="balanced_subsample", random_state=seed))
        thr = calibrate_thr(clf.predict_proba(X[va])[:, 1], (y[va] == 0) & (~is_atk[va]))
        dte = w.iloc[te].copy(); dte["pred"] = (clf.predict_proba(X[te])[:, 1] > thr).astype(int)
        atk_te = sorted(k for k in tek if k in atk); ben_te = sorted(k for k in tek if k not in atk)

        for cname, cfg in CONFIGS.items():
            pol0 = {}
            for k, g in dte.groupby("api_key"):
                g = g.sort_values("win_start")
                pol0[k] = simulate_entity_policy_v2(g["win_start"].astype(float).tolist(), g["pred"].tolist(), cfg)
            for d in DELAYS:
                pol = {k: shift_policy(v, d) for k, v in pol0.items()}
                b_thr = np.mean([any(s[1] != "normal" for s in pol[k]) for k in ben_te])
                b_rev = np.mean([any(s[1] == "review_queue" for s in pol[k]) for k in ben_te])
                b_adm = np.mean([s[2] for k in ben_te for s in pol[k]])
                adm = {k: admitted(q_all[q_all.api_key == k], pol[k],
                                   np.random.RandomState(seed + zlib.crc32(k.encode()) % 100000))
                       for k in atk_te}
                q2b = [first_action_queries(q_all[q_all.api_key == k], pol[k]) for k in atk_te]
                never = np.mean([v is None for v in q2b]); q2b = [v for v in q2b if v is not None]
                rows[(cname, d)].append({
                    "BenignEverThrottled": b_thr, "BenignEverReview": b_rev, "BenignMeanAdmit": b_adm,
                    "AtkMeanAdmit": np.mean([s[2] for k in atk_te for s in pol[k]]),
                    "AtkNeverMitigated": never,
                    "AtkQueriesToFirstAction": float(np.mean(q2b)) if q2b else np.nan,
                    "AtkAdmittedQ": float(np.mean([len(adm[k]) for k in atk_te])),
                    "AtkMSE": float(np.mean([surrogate_mse(adm[k], X_test, y_test, base_mse) for k in atk_te]))})
        print(f"[seed={seed}] done")

    spec = [("BenignEverThrottled", "pct"), ("BenignEverReview", "pct"), ("BenignMeanAdmit", "pct"),
            ("AtkMeanAdmit", "pct"), ("AtkNeverMitigated", "pct"),
            ("AtkQueriesToFirstAction", "raw"), ("AtkAdmittedQ", "raw"), ("AtkMSE", "mse")]
    hdr = f"{'Config | delay':<40}| " + " | ".join(f"{c:<17}" for c, _ in spec)
    print("\n" + "=" * len(hdr)); print(hdr); print("-" * len(hdr))
    out = {}
    for (c, d), rs in rows.items():
        s = summarize(rs, [c2 for c2, _ in spec])
        print(f"{c + ' | d=' + str(d):<40}| " + " | ".join(f"{cell(s[c2], k):<17}" for c2, k in spec))
        out[f"{c}|delay={d}"] = {c2: {"mean": s[c2][0], "std": s[c2][1], "n": s[c2][2]} for c2, _ in spec}
    save_json("mitigation_v10_delay.json", out)

if __name__ == "__main__":
    main()