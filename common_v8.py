# common_v8.py
# [v8-fix] 
import os, json, hashlib
import numpy as np
import pandas as pd

TARGET_FPR = 0.01
SEEDS = list(range(42, 52))
OUT_DIR = "results_v8"
os.makedirs(OUT_DIR, exist_ok=True)

KINDS = ("norm", "batch", "monitor", "trial",
         "atk_s0", "atk_s1", "atk_s2", "atk_s3", "atk_s4", "atk_s5", "atk_s6")

PATTERNS_ALL = {"S0": "S0_", "S1": "S1_", "S2": "S2_", "S3": "S3_",
                "S4": "S4_", "S5": "S5_", "S6": "S6_"}

def kind_of(k):
    k = str(k)
    for p in KINDS:
        if k.startswith(p): return p
    return "other"

def hash_key(k):
    return "unknown" if pd.isna(k) else hashlib.sha256(str(k).encode()).hexdigest()[:16]

def hash_keys(df_raw, key_col="api_key"):
  
    keys = df_raw[key_col].astype(str).unique()
    mp = {k: hash_key(k) for k in keys}
    kind = {mp[k]: kind_of(k) for k in keys}
    atk_raw = df_raw.loc[df_raw["label"].astype(str).str.lower() == "attack", key_col].astype(str).unique()
    atk = {mp[k] for k in atk_raw}
    df = df_raw.copy()
    df[key_col] = df[key_col].astype(str).map(mp)
    return df, kind, atk

def to_epoch_s(s):
   
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.astype("int64").to_numpy().astype(float) / 1e9
    num = pd.to_numeric(s, errors="coerce")
    if num.isna().mean() > 0.5:                      
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        return dt.astype("int64").to_numpy().astype(float) / 1e9
    return num.to_numpy(float)

def _split_group(items, seed):
    rng = np.random.RandomState(seed)
    items = sorted(items); rng.shuffle(items)
    n = len(items)
    if n < 3: return set(items), set(), set()
    n_te = max(1, int(round(0.2 * n)))
    n_va = max(1, int(round(0.1 * n))) if n >= 5 else 0
    return set(items[n_te + n_va:]), set(items[n_te:n_te + n_va]), set(items[:n_te])

def stratified_entity_split(kind_map, seed, holdout_kinds=()):
 
    tr, va, te = set(), set(), set()
    by_kind = {}
    for k, kd in kind_map.items():
        by_kind.setdefault(kd, []).append(k)
    for kd, ks in by_kind.items():
        if kd in holdout_kinds:
            te |= set(ks); continue
        a, b, c = _split_group(ks, seed)
        tr |= a; va |= b; te |= c
    return tr, va, te

def calibrate_thr(p_val, neg_mask, target=TARGET_FPR):
    if neg_mask.sum() == 0: return 0.5
    try:
        return float(np.quantile(p_val[neg_mask], 1 - target, method="higher")) + 1e-6
    except TypeError:
        return float(np.quantile(p_val[neg_mask], 1 - target, interpolation="higher")) + 1e-6

def dedup_alerts(df, group_col="api_key", cooldown_s=600):
    n = 0
    for _, g in df[df["pred"] == 1].groupby(group_col):
        last = -1e18
        for t in np.sort(g["win_start"].to_numpy()):
            if t - last >= cooldown_s:
                n += 1; last = t
    return n

def ent_fpr_k(d, k=3, window_s=1800, group_col="api_key"):
   
    ben = d[~d["is_atk"].astype(bool)]
    n_ben = ben[group_col].nunique()
    if n_ben == 0:
        return 0.0
    hit = 0
    for _, g in ben.groupby(group_col):
        t = np.sort(g.loc[g["pred"] == 1, "win_start"].to_numpy(dtype=float))
        if len(t) >= k and np.any(t[k - 1:] - t[:-(k - 1)] <= window_s):
            hit += 1
    return hit / n_ben

def compute_metrics(df_test, pred, is_atk_te, patterns=PATTERNS_ALL, group_col="api_key"):
    d = df_test.copy()
    d["pred"] = np.asarray(pred).astype(int)
    d["is_atk"] = np.asarray(is_atk_te).astype(bool)
    out = {}
    for lab, pat in patterns.items():
        m = d["type_mode"].astype(str).str.contains(pat) & (d["y_binary"] == 1)
        out[lab] = float(d.loc[m, "pred"].mean()) if m.sum() else np.nan
    v = [out[k] for k in patterns if not np.isnan(out[k])]
    out["Macro"] = float(np.mean(v)) if v else np.nan
    neg = (d["y_binary"] == 0) & (~d["is_atk"])
    out["WinFPR"] = float(d.loc[neg, "pred"].mean()) if neg.sum() else np.nan
    nk = d.loc[~d["is_atk"], group_col].unique()
    fk = d.loc[(~d["is_atk"]) & (d["pred"] == 1), group_col].unique()
    out["EntFPR"] = len(fk) / max(len(nk), 1)
    out["EntFPR3"] = ent_fpr_k(d, group_col=group_col)
    hours = max((d["win_start"].max() - d["win_start"].min()) / 3600.0, 1e-5)
    out["AlertHr"] = dedup_alerts(d[~d["is_atk"]], group_col) / hours
    return out

def summarize(rows, cols):
    s = {}
    for c in cols:
        a = np.array([r.get(c, np.nan) for r in rows], dtype=float)
        ok = a[~np.isnan(a)]
        s[c] = (float(ok.mean()), float(ok.std()), int(len(ok))) if len(ok) else (np.nan, np.nan, 0)
    return s

def fmt(cell, pct=True):
    m, s, n = cell
    if n == 0: return "n/a"
    return f"{m*100:5.1f}±{s*100:4.1f}({n})" if pct else f"{m:6.2f}±{s:5.2f}({n})"

def print_table(res, cols, name_w=30):
    hdr = f"{'Variant':<{name_w}}| " + " | ".join(f"{c:<15}" for c in cols)
    print("\n" + "=" * len(hdr)); print(hdr); print("-" * len(hdr))
    summary = {}
    for v, rows in res.items():
        s = summarize(rows, cols); summary[v] = {c: {"mean": s[c][0], "std": s[c][1], "n": s[c][2]} for c in cols}
        cells = [fmt(s[c], pct=(c != "AlertHr")) for c in cols]
        print(f"{v:<{name_w}}| " + " | ".join(f"{x:<15}" for x in cells))
    print("括号内为该列有测试数据的种子数；Macro = 测试集中出现的攻击策略 TPR 的无权平均")
    return summary

def save_json(name, obj):
    with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"[*] saved {OUT_DIR}/{name}")