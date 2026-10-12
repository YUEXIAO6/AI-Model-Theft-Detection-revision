
import argparse
import numpy as np, pandas as pd
from spp_experiments_v5 import SPPConfig, ModelConfig, train_rf, select_feature_columns, build_spp_context_features
from credit_data_v8 import prepare_credit
from common_v8 import *
from mit_state_machine_v2 import MitigationConfigV2, simulate_entity_policy_v2

SPP = SPPConfig(enabled=True, levels=(1, 2), lookback_windows=20, context="past")
CONFIGS = {
    "Aggressive (1,3) review=5%":   MitigationConfigV2(1, 3, 1800, 0.2, 0.05),
    "Conservative (3,5) review=5%": MitigationConfigV2(3, 5, 1800, 0.2, 0.05),
}
BENIGN_KINDS = ["norm", "batch", "monitor"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10); a = ap.parse_args()
    df_raw, w, kind_map, atk, _, _ = prepare_credit()
    y = w["y_binary"].to_numpy(int); is_atk = w["is_atk"].to_numpy()
    cols = select_feature_columns(w, "ours_all"); Xb = w[cols].to_numpy(float)


    ent    = {c: {k: {"n": 0, "thr": 0, "rev": 0} for k in BENIGN_KINDS} for c in CONFIGS}
    admit  = {c: {k: [] for k in BENIGN_KINDS} for c in CONFIGS}
    win    = {k: {"n": 0, "fp": 0} for k in BENIGN_KINDS}     
    alert3 = {k: {"n": 0, "hit": 0} for k in BENIGN_KINDS}    

    for seed in SEEDS[:a.seeds]:
        trk, vak, tek = stratified_entity_split(kind_map, seed)
        idx = lambda S: w.index[w["api_key"].isin(S)].to_numpy(); tr, va, te = idx(trk), idx(vak), idx(tek)
        Xs, _ = build_spp_context_features(w, cols, "api_key", "win_start", SPP, tr); X = np.hstack([Xb, Xs])
        clf = train_rf(X[tr], y[tr], ModelConfig(class_weight="balanced_subsample", random_state=seed))
        thr = calibrate_thr(clf.predict_proba(X[va])[:, 1], (y[va] == 0) & (~is_atk[va]))
        dte = w.iloc[te].copy(); dte["pred"] = (clf.predict_proba(X[te])[:, 1] > thr).astype(int)
        dte["kind"] = dte["api_key"].map(kind_map)
        ben = dte[~dte["is_atk"]]

        for kd in BENIGN_KINDS:
            g = ben[(ben["kind"] == kd) & (ben["y_binary"] == 0)]
            win[kd]["n"] += len(g); win[kd]["fp"] += int(g["pred"].sum())
            for k, gk in ben[ben["kind"] == kd].groupby("api_key"):
                t = np.sort(gk.loc[gk["pred"] == 1, "win_start"].to_numpy(float))
                alert3[kd]["n"] += 1
                if len(t) >= 3 and np.any(t[2:] - t[:-2] <= 1800): alert3[kd]["hit"] += 1

        for cname, cfg in CONFIGS.items():
            for k, gk in ben.groupby("api_key"):
                kd = kind_map[k]
                if kd not in BENIGN_KINDS: continue
                gk = gk.sort_values("win_start")
                pol = simulate_entity_policy_v2(gk["win_start"].astype(float).tolist(), gk["pred"].tolist(), cfg)
                ent[cname][kd]["n"] += 1
                ent[cname][kd]["thr"] += int(any(s[1] != "normal" for s in pol))
                ent[cname][kd]["rev"] += int(any(s[1] == "review_queue" for s in pol))
                admit[cname][kd] += [s[2] for s in pol]
        print(f"[seed={seed}] benign test entities by kind: {ben.groupby('kind')['api_key'].nunique().to_dict()}")

    print("\n=== Detector-level false positives by benign kind (pooled over seeds) ===")
    print(f"{'kind':<10}{'#neg windows':>14}{'WinFPR':>10}{'#entity-seeds':>15}{'EntFPR3(30min)':>16}")
    for kd in BENIGN_KINDS:
        print(f"{kd:<10}{win[kd]['n']:>14}{win[kd]['fp']/max(win[kd]['n'],1)*100:>9.2f}%"
              f"{alert3[kd]['n']:>15}{alert3[kd]['hit']/max(alert3[kd]['n'],1)*100:>15.1f}%")

    out = {"window_fpr": win, "alert3": alert3, "policy": {}}
    for cname in CONFIGS:
        print(f"\n=== {cname}: mitigation cost on benign entities by kind (pooled) ===")
        print(f"{'kind':<10}{'#entity-seeds':>15}{'EverThrottled':>15}{'EverReview':>12}{'MeanAdmit':>11}")
        out["policy"][cname] = {}
        for kd in BENIGN_KINDS:
            e = ent[cname][kd]; n = max(e["n"], 1)
            ma = float(np.mean(admit[cname][kd])) if admit[cname][kd] else 1.0
            print(f"{kd:<10}{e['n']:>15}{e['thr']/n*100:>14.1f}%{e['rev']/n*100:>11.1f}%{ma*100:>10.1f}%")
            out["policy"][cname][kd] = {"n": e["n"], "ever_throttled": e["thr"]/n, "ever_review": e["rev"]/n, "mean_admit": ma}
    save_json("mit_benign_by_kind_v8.json", out)

if __name__ == "__main__":
    main()