# export_share_v12.py
import json, numpy as np, pandas as pd
from credit_data_v8 import prepare_credit
df_raw, w, *_ = prepare_credit("traffic_logs.csv")
df = pd.read_csv("traffic_logs.csv")
t0 = df.timestamp.min()
df["bin"] = ((df.timestamp - t0) // 300).astype(int)
df["atk_credit"] = df.label.astype(str).str.lower().eq("attack") & df.endpoint.eq("/api/v1/credit_score")
req = df.groupby("bin")["atk_credit"].mean()
w["bin"] = ((w.win_start - w.win_start.min()) // 300).astype(int)
win = w[~w.is_atk | (w.y_binary == 1) | w.is_atk].groupby("bin")["y_binary"].mean()
bins = sorted(set(req.index) | set(win.index))
data = [{"min": b * 5, "req": float(req.get(b, 0)), "win": float(win.get(b, 0))} for b in bins]
print(json.dumps(data))