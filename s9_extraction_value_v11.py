
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import common_v11
from common_v8 import save_json
from mit_eval_v8 import target_model, surrogate_mse

df = pd.read_csv("traffic_logs_v10b.csv")
df = df[df.endpoint == "/api/v1/credit_score"].copy()
df["age"] = pd.to_numeric(df.age, errors="coerce"); df["income"] = pd.to_numeric(df.income, errors="coerce")
df = df.dropna(subset=["age", "income"]).sort_values("timestamp")
rng = np.random.RandomState(123)
Xt = np.column_stack((rng.uniform(18, 90, 1000), rng.uniform(10000, 120000, 1000)))
yt = target_model(Xt[:, 0], Xt[:, 1]); base = float(np.mean((yt - 0.5) ** 2))
KS = [25, 50, 100, 200, 400]
TYPES = ["S1_TimingMimic", "S9_StratJitter", "S5_DistMatch"]
out = {}
print(f"no-knowledge MSE = {base:.4f}")
print(f"{'type':<16}" + "".join(f"{'k='+str(k):>16}" for k in KS))
for t in TYPES:
    sub = df[df.type == t]; res = {k: [] for k in KS}
    for key, g in sub.groupby("api_key"):
        for k in KS:
            if len(g) >= k:
                res[k].append(surrogate_mse(g.iloc[:k], Xt, yt, base))
    print(f"{t:<16}" + "".join(f"{np.mean(res[k]):>9.4f}±{np.std(res[k]):.4f}" for k in KS))
    out[t] = {str(k): {"mean": float(np.mean(res[k])), "std": float(np.std(res[k])), "n": len(res[k])} for k in KS}
save_json("s9_extraction_value.json", out)