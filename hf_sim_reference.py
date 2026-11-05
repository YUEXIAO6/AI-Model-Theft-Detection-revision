# hf_sim_reference.py -- print simulator statistics; use them to set the HF collection parameters
import numpy as np
from hf_common import load_sim_samples

for scope in ("all", "credit"):
    s = load_sim_samples("traffic_logs.csv", scope)
    print(f"\n=== IAT scope = {scope} ===")
    for pat in ("steady", "burst"):
        for f in ("iat", "payload", "latency"):
            a = s[pat][f]
            print(f"{pat:<7}{f:<9} n={len(a):>6}  mean={a.mean():8.4f}  std={a.std():8.4f}  median={np.median(a):8.4f}")