# prada_baseline_v8.py
# PRADA (Juuti et al., EuroS&P 2019) 的 L7 日志复现，作为同数据源被动检测基线。
# 简化说明：(1) 特征为 (age, income) 经训练集正常用户 min-max 归一化；
#           (2) D 的加入规则 d_min > max(mean(D)-std(D), 0)；
#           (3) |D| >= min_d 后计算 W；score = 1 - W；窗口得分取窗口结束前最新的 score。
# 两种判定规则并列报告：
#   - "1%FPR"  : 与其它方法相同协议，val 正常窗口 99 分位阈值
#   - "native δ": PRADA 原文做法，δ 取使训练集良性用户全部不被标记的边界（thr = 训练良性 key 最高分）
# [v8-fix] timestamp 统一转 epoch 秒；第一个 seed 输出按种类的 PRADA 状态诊断与两种阈值。
import argparse, warnings
import numpy as np, pandas as pd
from scipy.stats import shapiro
from credit_data_v8 import prepare_credit
from common_v8 import *

COLS = list(PATTERNS_ALL) + ["Macro", "WinFPR", "EntFPR", "EntFPR3", "AlertHr"]

def prada_scores(q, ref_keys, min_d):
    ref = q[q["api_key"].isin(ref_keys)][["age", "income"]].to_numpy(float)
    if len(ref) == 0:
        ref = q[["age", "income"]].to_numpy(float)
    lo, hi = ref.min(0), ref.max(0); span = np.where(hi > lo, hi - lo, 1.0)
    out, state = {}, {}
    for key, g in q.sort_values("timestamp").groupby("api_key"):
        X = (g[["age", "income"]].to_numpy(float) - lo) / span
        ts = g["timestamp"].to_numpy(float)          # 已在 main() 里转成 epoch 秒
        P, D, W_last = [], [], None
        sc = np.zeros(len(g))
        for i, x in enumerate(X):
            if P:
                dmin = float(np.min(np.linalg.norm(np.asarray(P) - x, axis=1)))
                if not D: D.append(dmin)
                else:
                    T = max(np.mean(D) - np.std(D), 0.0)
                    if dmin > T: D.append(dmin)
                if len(D) >= min_d:
                    arr = np.asarray(D[-5000:])
                    if arr.std() < 1e-12: W_last = 0.0          # 完全重复的步长 → 极端非正态
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore"); W_last = float(shapiro(arr)[0])
            P.append(x)
            sc[i] = (1.0 - W_last) if W_last is not None else 0.0
        out[key] = (ts, sc)
        state[key] = {"n_q": len(g), "n_D": len(D),
                      "W": (W_last if W_last is not None else np.nan),
                      "max_score": float(sc.max())}
    return out, state

def window_scores(w, sc, window_s=60.0):
    s = np.zeros(len(w))
    for key, g in w.groupby("api_key"):
        if key not in sc: continue
        ts, scores = sc[key]
        pos = np.searchsorted(ts, g["win_start"].to_numpy(float) + window_s, side="right") - 1
        s[g.index.to_numpy()] = np.where(pos >= 0, scores[np.clip(pos, 0, None)], 0.0)
    return s

def diag_state(state, kind_map, split_of, min_d):
    rows = [{"key": k, "kind": kind_map.get(k, "?"), "split": split_of.get(k, "-"), **st}
            for k, st in state.items()]
    d = pd.DataFrame(rows); d["reached"] = d["W"].notna()
    g = d.groupby("kind").agg(n_keys=("key", "size"), reached_min_d=("reached", "sum"),
                              med_queries=("n_q", "median"), med_absD=("n_D", "median"),
                              med_W=("W", "median"), max_score=("max_score", "max"))
    print(f"\n[diag] per-kind PRADA state (min_d={min_d}); W 越低越像攻击, score=1-W")
    print(g.to_string(float_format=lambda x: f"{x:.3f}"))
    dd = d[d["split"].isin(["val", "test"]) & d["reached"]].sort_values(["split", "kind"])
    print(f"[diag] val/test keys that reached |D|>={min_d}: {len(dd)}")
    if len(dd):
        print(dd[["kind", "split", "n_q", "n_D", "W", "max_score"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="traffic_logs.csv")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--holdout", default="")
    ap.add_argument("--min_d", default="10,100")   # PRADA 原文在 ~100 次查询后才判定；10 是宽松版
    a = ap.parse_args()
    holdout = tuple(x for x in a.holdout.split(",") if x)
    min_ds = [int(x) for x in a.min_d.split(",")]

    df_raw, w, kind_map, atk, _, _ = prepare_credit(a.csv)
    q = df_raw[df_raw["endpoint"] == "/api/v1/credit_score"].copy()
    if len(q) == 0:
        print("[diag] endpoint 过滤后为空！df_raw.endpoint 取值如下：")
        print(df_raw["endpoint"].value_counts().to_string()); return
    q["age"] = pd.to_numeric(q["age"], errors="coerce"); q["income"] = pd.to_numeric(q["income"], errors="coerce")
    q = q.dropna(subset=["age", "income"])
    q["timestamp"] = to_epoch_s(q["timestamp"]); q = q.dropna(subset=["timestamp"])
    y = w["y_binary"].to_numpy(int); is_atk = w["is_atk"].to_numpy()
    print(f"[diag] queries={len(q)} keys={q.api_key.nunique()} attackers_in_q={q.api_key.isin(atk).sum()>0} "
          f"ts range=({q.timestamp.min():.0f},{q.timestamp.max():.0f}) "
          f"win_start range=({w.win_start.min():.0f},{w.win_start.max():.0f})")

    names = {}
    for m in min_ds:
        names[(m, "cal")] = f"PRADA |D|>={m} (1%FPR)"
        names[(m, "nat")] = f"PRADA |D|>={m} (native δ)"
    res = {n: [] for n in names.values()}
    first = True
    for seed in SEEDS[:a.seeds]:
        trk, vak, tek = stratified_entity_split(kind_map, seed, holdout)
        idx = lambda S: w.index[w["api_key"].isin(S)].to_numpy()
        va, te = idx(vak), idx(tek)
        ref_keys = {k for k in trk if k not in atk}
        split_of = {**{k: "train" for k in trk}, **{k: "val" for k in vak}, **{k: "test" for k in tek}}
        for m in min_ds:
            sc, state = prada_scores(q, ref_keys, m)
            s = window_scores(w, sc)
            thr_cal = calibrate_thr(s[va], (y[va] == 0) & (~is_atk[va]))
            thr_nat = max([state[k]["max_score"] for k in ref_keys if k in state], default=0.0) + 1e-6
            if first:
                neg = (y == 0) & (~is_atk)
                print(f"\n[diag] min_d={m}: nonzero window scores={int((s>0).sum())}/{len(s)}, "
                      f"attack-window mean={s[y==1].mean():.3f}, normal={s[neg].mean():.3f}, "
                      f"test max attack={s[te][y[te]==1].max() if (y[te]==1).any() else float('nan'):.3f}, "
                      f"test max normal={s[te][neg[te]].max() if neg[te].any() else float('nan'):.3f}")
                print(f"[diag] thr(1%FPR on val)={thr_cal:.4f} | thr(native δ, max train-benign score)={thr_nat:.4f}")
                diag_state(state, kind_map, split_of, m)
            res[names[(m, "cal")]].append(compute_metrics(w.iloc[te], (s[te] > thr_cal).astype(int), is_atk[te]))
            res[names[(m, "nat")]].append(compute_metrics(w.iloc[te], (s[te] > thr_nat).astype(int), is_atk[te]))
        first = False
        print(f"[seed={seed}] done")
    summary = print_table(res, COLS)
    save_json(f"prada_v8_{a.csv.replace('.csv','')}.json",
              {"summary": summary, "per_seed": res, "holdout": holdout, "min_d": min_ds,
               "rules": {"1%FPR": "val normal-window 99th percentile (same protocol as other methods)",
                         "native δ": "PRADA-style: threshold = max score of any training benign key"}})

if __name__ == "__main__":
    main()