# prada_relaxed_v8.py —— PRADA 放宽变体：所有 d_min 无条件加入 D（去掉 mean-std 门控）。
# 归一化 / Shapiro-Wilk / 窗口取分 / 两套阈值 / 诊断输出 与 prada_baseline_v8.py 完全一致，
# 用于排除“PRADA 失效仅因 D 集合不增长这一实现细节”的质疑。
import argparse, warnings
import numpy as np, pandas as pd
from scipy.stats import shapiro
from credit_data_v8 import prepare_credit
from common_v8 import *
from prada_baseline_v8 import window_scores, diag_state   # 只复用函数，main 有 __name__ 保护

COLS = list(PATTERNS_ALL) + ["Macro", "WinFPR", "EntFPR", "EntFPR3", "AlertHr"]

def prada_scores_relaxed(q, ref_keys, min_d, cap=5000):
    ref = q[q["api_key"].isin(ref_keys)][["age", "income"]].to_numpy(float)
    if len(ref) == 0: ref = q[["age", "income"]].to_numpy(float)
    lo, hi = ref.min(0), ref.max(0); span = np.where(hi > lo, hi - lo, 1.0)
    out, state = {}, {}
    for key, g in q.sort_values("timestamp").groupby("api_key"):
        X = (g[["age", "income"]].to_numpy(float) - lo) / span
        ts = g["timestamp"].to_numpy(float)
        P, D, W_last = [], [], None
        sc = np.zeros(len(g))
        for i, x in enumerate(X):
            if P:
                D.append(float(np.min(np.linalg.norm(np.asarray(P) - x, axis=1))))   # 【唯一改动】无条件入集
                if len(D) >= min_d:
                    arr = np.asarray(D[-cap:])
                    if arr.std() < 1e-12: W_last = 0.0
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore"); W_last = float(shapiro(arr)[0])
            P.append(x)
            sc[i] = (1.0 - W_last) if W_last is not None else 0.0
        out[key] = (ts, sc)
        state[key] = {"n_q": len(g), "n_D": len(D),
                      "W": (W_last if W_last is not None else np.nan), "max_score": float(sc.max())}
    return out, state

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="traffic_logs.csv")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--holdout", default="")
    ap.add_argument("--min_d", default="10,30,100")
    a = ap.parse_args()
    holdout = tuple(x for x in a.holdout.split(",") if x); min_ds = [int(x) for x in a.min_d.split(",")]

    df_raw, w, kind_map, atk, _, _ = prepare_credit(a.csv)
    q = df_raw[df_raw["endpoint"] == "/api/v1/credit_score"].copy()
    q["age"] = pd.to_numeric(q["age"], errors="coerce"); q["income"] = pd.to_numeric(q["income"], errors="coerce")
    q = q.dropna(subset=["age", "income"])
    q["timestamp"] = to_epoch_s(q["timestamp"]); q = q.dropna(subset=["timestamp"])
    y = w["y_binary"].to_numpy(int); is_atk = w["is_atk"].to_numpy()

    names = {(m, r): f"PRADA-relaxed |D|>={m} ({'1%FPR' if r == 'cal' else 'native δ'})"
             for m in min_ds for r in ("cal", "nat")}
    res = {n: [] for n in names.values()}
    first = True
    for seed in SEEDS[:a.seeds]:
        trk, vak, tek = stratified_entity_split(kind_map, seed, holdout)
        idx = lambda S: w.index[w["api_key"].isin(S)].to_numpy(); va, te = idx(vak), idx(tek)
        ref_keys = {k for k in trk if k not in atk}
        split_of = {**{k: "train" for k in trk}, **{k: "val" for k in vak}, **{k: "test" for k in tek}}
        for m in min_ds:
            sc, state = prada_scores_relaxed(q, ref_keys, m)
            s = window_scores(w, sc)
            thr_cal = calibrate_thr(s[va], (y[va] == 0) & (~is_atk[va]))
            thr_nat = max([state[k]["max_score"] for k in ref_keys if k in state], default=0.0) + 1e-6
            if first:
                neg = (y == 0) & (~is_atk)
                print(f"\n[diag] relaxed min_d={m}: nonzero window scores={int((s > 0).sum())}/{len(s)} | "
                      f"attack-window mean={s[y == 1].mean():.3f} normal={s[neg].mean():.3f} | "
                      f"thr_cal={thr_cal:.4f} thr_nat={thr_nat:.4f}")
                diag_state(state, kind_map, split_of, m)
            res[names[(m, "cal")]].append(compute_metrics(w.iloc[te], (s[te] > thr_cal).astype(int), is_atk[te]))
            res[names[(m, "nat")]].append(compute_metrics(w.iloc[te], (s[te] > thr_nat).astype(int), is_atk[te]))
        first = False
        print(f"[seed={seed}] done")
    summary = print_table(res, COLS, name_w=40)
    save_json(f"prada_relaxed_v8_{a.csv.replace('.csv', '')}.json",
              {"summary": summary, "per_seed": res, "holdout": holdout, "min_d": min_ds,
               "rule": "every d_min appended to D unconditionally (mean-std gate removed)"})

if __name__ == "__main__":
    main()