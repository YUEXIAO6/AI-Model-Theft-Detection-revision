
import argparse, os, json
import numpy as np, pandas as pd
from spp_experiments_v5 import (ExperimentConfig, WindowConfig, SPPConfig, ModelConfig,
    load_and_prepare_df, build_windows_df, train_rf, select_feature_columns,
    build_spp_context_features, add_window_id)
from window_relabel_credit_only import relabel_windows_credit_only
from common_v8 import kind_of, calibrate_thr, PATTERNS_ALL, save_json

PERSIST = ("norm", "batch", "monitor")
COHORT = ["n_keys", "new_key_frac"]

def split3(items, seed):
    rng = np.random.RandomState(seed); items = sorted(items); rng.shuffle(items)
    n = len(items)
    n_te = max(1, int(round(0.2 * n))) if n >= 3 else 0
    n_va = max(1, int(round(0.1 * n))) if n >= 5 else 0
    return set(items[n_te + n_va:]), set(items[n_te:n_te + n_va]), set(items[:n_te])

def build_key_windows(df, cfg, win_cfg):
    w = build_windows_df(df, cfg, win_cfg, True, True)
    w = relabel_windows_credit_only(df_raw=df, windows_df=w, window_s=win_cfg.window_s,
                                    group_col=cfg.group_col, credit_endpoint="/api/v1/credit_score")
    return w.sort_values("win_start").reset_index(drop=True)

def fit_scores(win, feats, tr, val, te, spp_cfg, seed):
    Xb = win[feats].to_numpy(float)
    Xs, _ = build_spp_context_features(win, feats, "api_key", "win_start", spp_cfg, tr)
    X = np.hstack([Xb, Xs]); y = win["y_binary"].to_numpy(int)
    clf = train_rf(X[tr], y[tr], ModelConfig(class_weight="balanced_subsample", random_state=seed))
    return clf.predict_proba(X[val])[:, 1], clf.predict_proba(X[te])[:, 1]

def ip_flag(dfk, ip_te, p, thr):
    t = pd.DataFrame({"ip": ip_te["api_key"].values, "win_id": ip_te["win_id"].values, "f": (p > thr).astype(int)})
    return dfk.merge(t, on=["ip", "win_id"], how="left")["f"].fillna(0).to_numpy(int)

def metrics(dfk, pred):
    d = dfk.copy(); d["pred"] = pred.astype(int)
    normal = (d["y_binary"] == 0) & (~d["is_atk"])
    out = {"WinFPR": float(d.loc[normal, "pred"].mean()) if normal.sum() else np.nan}
    pk = d.loc[d["persist"], "api_key"].unique(); fk = d.loc[d["persist"] & (d["pred"] == 1), "api_key"].unique()
    out["PersistEntFPR"] = len(fk) / max(len(pk), 1)
    for lab, pat in PATTERNS_ALL.items():
        m = d["type_mode"].astype(str).str.contains(pat) & (d["y_binary"] == 1)
        out[lab] = float(d.loc[m, "pred"].mean()) if m.sum() else np.nan
    v = [out[k] for k in PATTERNS_ALL if not np.isnan(out[k])]
    out["Macro"] = float(np.mean(v)) if v else np.nan
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="traffic_logs_v7_2.csv")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--holdout", default="atk_s4,atk_s5,atk_s6")
    ap.add_argument("--out", default="granularity_v7_3.json")
    a = ap.parse_args()
    HOLD = tuple(x for x in a.holdout.split(",") if x)

    cfg = ExperimentConfig(csv_file=a.csv, group_col="api_key")
    win_cfg = WindowConfig(window_s=60, attack_ratio_thr=0.1)
    spp_cfg = SPPConfig(enabled=True, levels=(1, 2), lookback_windows=20, context="past")

    df_key = load_and_prepare_df(cfg); df_key["src_ip"] = df_key["src_ip"].astype(str)
    key2ip = df_key.groupby("api_key")["src_ip"].agg(lambda s: s.mode().iloc[0]).to_dict()

    win_key = build_key_windows(df_key, cfg, win_cfg)
    win_key["ip"] = win_key["api_key"].map(key2ip)
    win_key["kind"] = win_key["api_key"].map(kind_of)
    win_key["is_atk"] = win_key["api_key"].astype(str).str.startswith("atk")
    win_key["persist"] = win_key["kind"].isin(PERSIST)
    win_key["is_hold"] = win_key["kind"].isin(HOLD)

    df_ip = df_key.copy(); df_ip["orig_key"] = df_ip["api_key"]; df_ip["api_key"] = df_ip["src_ip"]
    win_ip = build_windows_df(df_ip, cfg, win_cfg, True, True)
    df_ip_w = add_window_id(df_ip, win_cfg, cfg)
    df_ip_w["is_hold"] = df_ip_w["orig_key"].map(lambda k: kind_of(k) in HOLD)

    lab = (win_key.groupby(["ip", "win_id"])["y_binary"].max().reset_index()
           .rename(columns={"ip": "api_key", "y_binary": "y_new"}))
    win_ip = win_ip.merge(lab, on=["api_key", "win_id"], how="left")
    win_ip["y_binary"] = win_ip["y_new"].fillna(0).astype(int); win_ip = win_ip.drop(columns=["y_new"])
    agg = df_ip_w.groupby(["api_key", "win_id"]).agg(atk_frac=("is_attack", "mean"), hold_frac=("is_hold", "mean")).reset_index()
    win_ip = win_ip.merge(agg, on=["api_key", "win_id"], how="left")
    nk = df_ip_w.groupby(["api_key", "win_id"])["orig_key"].nunique().rename("n_keys")
    first = df_ip_w.groupby("orig_key")["win_id"].min().rename("first_win")
    tmp = df_ip_w[["api_key", "win_id", "orig_key"]].drop_duplicates().merge(first, left_on="orig_key", right_index=True)
    tmp["is_new"] = (tmp["win_id"] == tmp["first_win"]).astype(float)
    nf = tmp.groupby(["api_key", "win_id"])["is_new"].mean().rename("new_key_frac")
    win_ip = win_ip.merge(pd.concat([nk, nf], axis=1).reset_index(), on=["api_key", "win_id"], how="left")
    win_ip[COHORT] = win_ip[COHORT].fillna(0)
    win_ip = win_ip.sort_values("win_start").reset_index(drop=True)

    feats_key = select_feature_columns(win_key, "ours_all"); feats_coh = feats_key + COHORT

    # 体检
    print("[*] keys by kind:\n" + pd.Series({k: kind_of(k) for k in key2ip}).value_counts().to_string())
    print(f"[*] HOLDOUT (test-only) kinds: {HOLD}")
    print(f"[*] key-windows={len(win_key)} ip-windows={len(win_ip)} | key-windows held out of train/val: {int(win_key.is_hold.sum())}")
    print(win_key.loc[win_key.y_binary == 1, "type_mode"].value_counts().to_string())

    sybil_ips = {ip for k, ip in key2ip.items() if kind_of(k) in ("atk_s4", "atk_s6")}
    atk_ips = {ip for k, ip in key2ip.items() if str(k).startswith("atk")}
    all_ips = set(key2ip.values())
    groups = [sorted(sybil_ips), sorted(atk_ips - sybil_ips), sorted(all_ips - atk_ips)]

    VARIANTS = ["Key (1%)", "IP base (1%)", "IP+cohort (1%)", "OR Key+IPbase (0.5%x2)", "OR Key+IPcohort (0.5%x2)"]
    COLS = list(PATTERNS_ALL) + ["Macro", "WinFPR", "PersistEntFPR"]
    res = {v: {c: [] for c in COLS} for v in VARIANTS}

    for seed in range(42, 42 + a.seeds):
        tr_s, va_s, te_s = set(), set(), set()
        for g in groups:
            t, v, e = split3(g, seed); tr_s |= t; va_s |= v; te_s |= e
        # Key 级：train/val 剔除 holdout 种类；test 全保留
        kt = win_key.index[win_key["ip"].isin(tr_s) & ~win_key["is_hold"]].to_numpy()
        kv = win_key.index[win_key["ip"].isin(va_s) & ~win_key["is_hold"]].to_numpy()
        ke = win_key.index[win_key["ip"].isin(te_s)].to_numpy()
        # IP 级：train/val 剔除含 holdout 请求的 IP 窗口
        it = win_ip.index[win_ip["api_key"].isin(tr_s) & (win_ip["hold_frac"] == 0)].to_numpy()
        iv = win_ip.index[win_ip["api_key"].isin(va_s) & (win_ip["hold_frac"] == 0)].to_numpy()
        ie = win_ip.index[win_ip["api_key"].isin(te_s)].to_numpy()

        pk_val, pk_te = fit_scores(win_key, feats_key, kt, kv, ke, spp_cfg, seed)
        pb_val, pb_te = fit_scores(win_ip, feats_key, it, iv, ie, spp_cfg, seed)
        pc_val, pc_te = fit_scores(win_ip, feats_coh, it, iv, ie, spp_cfg, seed)

        neg_kv = (win_key["y_binary"].to_numpy()[kv] == 0) & (~win_key["is_atk"].to_numpy()[kv])
        neg_iv = (win_ip["y_binary"].to_numpy()[iv] == 0) & (win_ip["atk_frac"].to_numpy()[iv] == 0)
        dfk = win_key.iloc[ke][["api_key", "ip", "win_id", "type_mode", "y_binary", "is_atk", "persist"]].reset_index(drop=True)
        ip_te = win_ip.iloc[ie][["api_key", "win_id"]].reset_index(drop=True)

        P = {}
        for tgt, tag in [(0.01, "1"), (0.005, "05")]:
            fk = (pk_te > calibrate_thr(pk_val, neg_kv, tgt)).astype(int)
            fb = ip_flag(dfk, ip_te, pb_te, calibrate_thr(pb_val, neg_iv, tgt))
            fc = ip_flag(dfk, ip_te, pc_te, calibrate_thr(pc_val, neg_iv, tgt))
            if tag == "1": P["Key (1%)"], P["IP base (1%)"], P["IP+cohort (1%)"] = fk, fb, fc
            else: P["OR Key+IPbase (0.5%x2)"], P["OR Key+IPcohort (0.5%x2)"] = np.maximum(fk, fb), np.maximum(fk, fc)
        for v in VARIANTS:
            m = metrics(dfk, P[v])
            for c in COLS: res[v][c].append(m[c])
        print(f"[seed={seed}] test IPs={len(te_s)} key-test windows={len(dfk)} ip-val-neg={int(neg_iv.sum())} key-val-neg={int(neg_kv.sum())}")

    hdr = f"{'Variant':<26}| " + " | ".join(f"{c:<14}" for c in COLS)
    print("\n" + "=" * len(hdr)); print(hdr); print("-" * len(hdr))
    summary = {}
    for v in VARIANTS:
        cells, summary[v] = [], {}
        for c in COLS:
            arr = np.array(res[v][c], dtype=float); ok = arr[~np.isnan(arr)]
            if len(ok):
                cells.append(f"{ok.mean()*100:5.1f}±{ok.std()*100:4.1f}({len(ok)})")
                summary[v][c] = {"mean": float(ok.mean()), "std": float(ok.std()), "n": int(len(ok))}
            else:
                cells.append("n/a"); summary[v][c] = {"mean": None, "std": None, "n": 0}
        print(f"{v:<26}| " + " | ".join(f"{x:<14}" for x in cells))
    print("S4/S5/S6 为 zero-day（未参与训练/校准）；PersistEntFPR 只统计 norm/batch/monitor 持久租户")
    save_json(a.out, {"summary": summary, "per_seed": res, "holdout": HOLD})

if __name__ == "__main__":
    main()