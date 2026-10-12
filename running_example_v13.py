# running_example_v13.py
import numpy as np, pandas as pd
from spp_experiments_v5 import SPPConfig, ModelConfig, train_rf, build_spp_context_features
from spp_experiments_v5_1 import select_feature_columns
from credit_data_v8 import prepare_credit
from common_v8 import *
from mit_eval_v8 import target_model, surrogate_mse

SEED = 42
df_raw, w, kind_map, atk, _, _ = prepare_credit()
cols = select_feature_columns(w, "ours_all")
trk, vak, tek = stratified_entity_split(kind_map, SEED)
idx = lambda S: w.index[w["api_key"].isin(S)].to_numpy()
tr, va, te = idx(trk), idx(vak), idx(tek)
spp = SPPConfig(True, (1, 2), 20, "past")
Xb = w[cols].to_numpy(float)
Xs, names = build_spp_context_features(w, cols, "api_key", "win_start", spp, tr)
X = np.hstack([Xb, Xs]); y = w["y_binary"].to_numpy(int); is_atk = w["is_atk"].to_numpy()
clf = train_rf(X[tr], y[tr], ModelConfig(class_weight="balanced_subsample", random_state=SEED))
p = clf.predict_proba(X)[:, 1]
thr = calibrate_thr(p[va], (y[va] == 0) & (~is_atk[va]))
print(f"threshold = {thr:.4f}")

all_names = cols + list(names)
def col(n): return X[:, all_names.index(n)]
w["score"] = p; w["flag"] = p > thr
w["spp_mean_pairs"] = col("spp_l1_seg0_mean_credit_unique_pairs")
w["spp_max_pairs"] = col("spp_l1_seg0_max_credit_unique_pairs")
w["spp_mean_ageent"] = col("spp_l1_seg0_mean_credit_age_entropy")

q = df_raw[df_raw.endpoint == "/api/v1/credit_score"].copy()
q["age"] = pd.to_numeric(q.age, errors="coerce"); q["income"] = pd.to_numeric(q.income, errors="coerce")
q["timestamp"] = to_epoch_s(q.timestamp); q = q.dropna(subset=["age", "income"]).sort_values("timestamp")
rng = np.random.RandomState(123)
Xt = np.column_stack((rng.uniform(18, 90, 1000), rng.uniform(10000, 120000, 1000)))
yt = target_model(Xt[:, 0], Xt[:, 1]); base = float(np.mean((yt - 0.5) ** 2))

def show(key, n=6, around_first_flag=True):
    g = w[w.api_key == key].sort_values("win_start").reset_index(drop=True)
    f = g.index[g.flag]
    start = max(int(f[0]) - 2, 0) if (around_first_flag and len(f)) else 5
    g = g.iloc[start:start + n].copy()
    qk = q[q.api_key == key]
    rows = []
    for _, r in g.iterrows():
        k = int((qk.timestamp < r.win_start + 60).sum())
        mse = surrogate_mse(qk.iloc[:k], Xt, yt, base) if k >= 5 else base
        rows.append({"win_idx": int(r.hist_idx) if "hist_idx" in r else "", "req_cnt": r.req_cnt,
                     "uniq_pairs": r.credit_unique_pairs, "age_entropy": round(r.credit_age_entropy, 2),
                     "SPP mean pairs": round(r.spp_mean_pairs, 2), "SPP max pairs": r.spp_max_pairs,
                     "score": round(r.score, 3), "flag": bool(r.flag),
                     "cum. credit queries": k, "surrogate MSE": round(mse, 4)})
    print(f"\n== {key} ({kind_map[key]}) ==\n", pd.DataFrame(rows).to_string(index=False))

s1 = sorted(k for k in tek if kind_map[k] == "atk_s1")[0]
nm = sorted(k for k in tek if kind_map[k] == "norm")[0]
show(s1); show(nm, around_first_flag=False)
print(f"\nno-knowledge MSE = {base:.4f}")