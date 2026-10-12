# mit_eval_v8.py —— 闭环缓解评估（10 种子）：攻击者代理模型 MSE + 良性用户代价 + 拦截前查询数
# [v8-fix] (a) MLPRegressor 改为关键字传参（新版 sklearn 第一个位置参数是 loss）；
#          (b) q_all.timestamp 统一转 epoch 秒，与 win_start 对齐。
import argparse
import numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from spp_experiments_v5 import SPPConfig, ModelConfig, train_rf, select_feature_columns, build_spp_context_features
from credit_data_v8 import prepare_credit
from common_v8 import *
from mit_state_machine_v2 import MitigationConfigV2, simulate_entity_policy_v2

SPP = SPPConfig(enabled=True, levels=(1, 2), lookback_windows=20, context="past")
CONFIGS = {
    "Aggressive (1,3) review=0%":    MitigationConfigV2(1, 3, 1800, 0.2, 0.0),    # = v1 语义
    "Aggressive (1,3) review=5%":    MitigationConfigV2(1, 3, 1800, 0.2, 0.05),
    "Conservative (3,5) review=5%":  MitigationConfigV2(3, 5, 1800, 0.2, 0.05),
}

def target_model(age, income):
    s = 0.4 * np.sin((age / 100) * np.pi) + 0.6 * (1 - np.exp(-(income / 150000) * 3))
    return np.round(np.clip(s, 0, 1), 4)

def fit_surrogate(X, y, seed=42):
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    if len(X) < 25:
        m = MLPRegressor(hidden_layer_sizes=(32, 16), solver="lbfgs", alpha=0.01,
                         max_iter=800, random_state=seed)
    else:
        m = MLPRegressor(hidden_layer_sizes=(64, 32), solver="adam", alpha=0.01,
                         max_iter=2000, early_stopping=True, validation_fraction=0.2,
                         n_iter_no_change=30, random_state=seed)
    m.fit(Xs, y); return sc, m

def surrogate_mse(q, X_test, y_test, base_mse):
    if len(q) < 5: return base_mse
    X = q[["age", "income"]].to_numpy(float); y = target_model(X[:, 0], X[:, 1])
    sc, m = fit_surrogate(X, y)
    return float(mean_squared_error(y_test, np.clip(m.predict(sc.transform(X_test)), 0, 1)))

def admitted(q, policy, rng, window_s=60.0):
    starts = np.array([p[0] for p in policy], dtype=float); rates = np.array([p[2] for p in policy], dtype=float)
    ts = q["timestamp"].to_numpy(float)
    pos = np.searchsorted(starts, ts, side="right") - 1
    ok = (pos >= 0) & (ts < starts[np.clip(pos, 0, None)] + window_s)
    r = np.where(ok, rates[np.clip(pos, 0, None)], 1.0)
    return q[rng.random_sample(len(q)) < r]

def first_action_queries(q, policy, window_s=60.0):
    ts = q["timestamp"].to_numpy(float)
    for t, st, _ in policy:
        if st != "normal": return int((ts <= t + window_s).sum())
    return None

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10); a = ap.parse_args()
    df_raw, w, kind_map, atk, _, _ = prepare_credit()
    y = w["y_binary"].to_numpy(int); is_atk = w["is_atk"].to_numpy()
    cols = select_feature_columns(w, "ours_all"); Xb = w[cols].to_numpy(float)
    q_all = df_raw[df_raw["endpoint"] == "/api/v1/credit_score"].copy()
    q_all["age"] = pd.to_numeric(q_all["age"], errors="coerce"); q_all["income"] = pd.to_numeric(q_all["income"], errors="coerce")
    q_all = q_all.dropna(subset=["age", "income"])
    # [v8-fix] 时间统一为 epoch 秒
    q_all["timestamp"] = to_epoch_s(q_all["timestamp"]); q_all = q_all.dropna(subset=["timestamp"])
    print(f"[diag] queries={len(q_all)} ts range=({q_all.timestamp.min():.0f},{q_all.timestamp.max():.0f}) "
          f"win_start range=({w.win_start.min():.0f},{w.win_start.max():.0f})")

    rng0 = np.random.RandomState(123)
    X_test = np.column_stack((rng0.uniform(18, 90, 1000), rng0.uniform(10000, 120000, 1000)))
    y_test = target_model(X_test[:, 0], X_test[:, 1]); base_mse = float(mean_squared_error(y_test, np.full_like(y_test, 0.5)))
    print(f"[*] no-knowledge baseline MSE = {base_mse:.4f}")

    rows = {c: [] for c in CONFIGS}; no_mit_mse = []
    for seed in SEEDS[:a.seeds]:
        trk, vak, tek = stratified_entity_split(kind_map, seed)
        idx = lambda S: w.index[w["api_key"].isin(S)].to_numpy(); tr, va, te = idx(trk), idx(vak), idx(tek)
        Xs, _ = build_spp_context_features(w, cols, "api_key", "win_start", SPP, tr); X = np.hstack([Xb, Xs])
        clf = train_rf(X[tr], y[tr], ModelConfig(class_weight="balanced_subsample", random_state=seed))
        thr = calibrate_thr(clf.predict_proba(X[va])[:, 1], (y[va] == 0) & (~is_atk[va]))
        dte = w.iloc[te].copy(); dte["pred"] = (clf.predict_proba(X[te])[:, 1] > thr).astype(int)

        atk_te = [k for k in tek if k in atk]; ben_te = [k for k in tek if k not in atk]
        no_mit_mse += [surrogate_mse(q_all[q_all.api_key == k], X_test, y_test, base_mse) for k in atk_te]

        for cname, cfg in CONFIGS.items():
            rng = np.random.RandomState(seed)
            pol = {}
            for k, g in dte.groupby("api_key"):
                g = g.sort_values("win_start")
                pol[k] = simulate_entity_policy_v2(g["win_start"].astype(float).tolist(), g["pred"].tolist(), cfg)
            # 良性代价
            b_thr = np.mean([any(s[1] != "normal" for s in pol[k]) for k in ben_te]) if ben_te else np.nan
            b_rev = np.mean([any(s[1] == "review_queue" for s in pol[k]) for k in ben_te]) if ben_te else np.nan
            b_admit = np.mean([s[2] for k in ben_te for s in pol[k]]) if ben_te else np.nan
            # 攻击者遏制
            q2b = [first_action_queries(q_all[q_all.api_key == k], pol[k]) for k in atk_te]
            never = np.mean([v is None for v in q2b]) if q2b else np.nan
            q2b = [v for v in q2b if v is not None]
            mit_mse = [surrogate_mse(admitted(q_all[q_all.api_key == k], pol[k], rng), X_test, y_test, base_mse) for k in atk_te]
            a_admit = np.mean([s[2] for k in atk_te for s in pol[k]]) if atk_te else np.nan
            rows[cname].append({"BenignEverThrottled": b_thr, "BenignEverReview": b_rev, "BenignMeanAdmit": b_admit,
                                "AtkMeanAdmit": a_admit, "AtkNeverMitigated": never,
                                "AtkQueriesToFirstAction": float(np.mean(q2b)) if q2b else np.nan,
                                "AtkMSE_mitigated": float(np.mean(mit_mse)) if mit_mse else np.nan})
        print(f"[seed={seed}] attackers in test={len(atk_te)} benign={len(ben_te)}")

    pct = ["BenignEverThrottled", "BenignEverReview", "BenignMeanAdmit", "AtkMeanAdmit", "AtkNeverMitigated"]
    raw = ["AtkQueriesToFirstAction", "AtkMSE_mitigated"]
    print(f"\nNo mitigation (detect only): attacker surrogate MSE = {np.mean(no_mit_mse):.4f} ± {np.std(no_mit_mse):.4f}  (baseline {base_mse:.4f})")
    hdr = f"{'Config':<30}| " + " | ".join(f"{c:<22}" for c in pct + raw)
    print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
    summary = {}
    for c, rs in rows.items():
        s = summarize(rs, pct + raw); summary[c] = {k: {"mean": s[k][0], "std": s[k][1], "n": s[k][2]} for k in s}
        print(f"{c:<30}| " + " | ".join(f"{fmt(s[k], pct=(k in pct)):<22}" for k in pct + raw))
    save_json("mitigation_v8.json", {"summary": summary, "per_seed": rows,
                                     "no_mitigation_mse": {"mean": float(np.mean(no_mit_mse)), "std": float(np.std(no_mit_mse))},
                                     "baseline_mse": base_mse})

if __name__ == "__main__":
    main()