# hf_common.py -- simulator-side samples for the HF sanity check
import numpy as np, pandas as pd

CREDIT = "/api/v1/credit_score"

def _per_key_diffs(d):
    out = []
    for _, g in d.groupby("api_key"):
        t = np.sort(g["timestamp"].to_numpy(float))
        if len(t) > 1:
            out.append(np.diff(t))
    return np.concatenate(out) if out else np.array([])

def load_sim_samples(csv="traffic_logs.csv", iat_scope="all"):
    """steady = Legit clients; burst = S0_Stealthy intra-burst (gaps < 1 s).
    iat_scope: 'all' = all requests of the key, 'credit' = credit_score requests only."""
    df = pd.read_csv(csv)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    out = {}

    leg = df[df["type"] == "Legit"]
    leg_c = leg[leg["endpoint"] == CREDIT]
    src = leg if iat_scope == "all" else leg_c
    out["steady"] = {"iat": _per_key_diffs(src),
                     "payload": leg_c["body_len"].to_numpy(float),
                     "latency": leg_c["proc_duration"].to_numpy(float)}

    s0 = df[df["type"] == "S0_Stealthy"]
    iat = _per_key_diffs(s0)
    out["burst"] = {"iat": iat[iat < 1.0],
                    "payload": s0["body_len"].to_numpy(float),
                    "latency": s0["proc_duration"].to_numpy(float)}
    return out