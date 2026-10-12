# make_tenant_log_v12.py
# --t N            : 每个攻击操作者控制 N 个租户（0 = 不聚合，即 key 级）
# --group_trials M : 把短命良性 trial key 归入 M 个企业租户（对照：良性租户也有多个 key）
import re, zlib, argparse
import pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--inp", default="traffic_logs_v12.csv")
ap.add_argument("--t", type=int, default=0)
ap.add_argument("--group_trials", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

R3 = re.compile(r"^(atk_s3)_op(\d+)_k(\d+)$")
R46 = re.compile(r"^(atk_s4|atk_s6)_op(\d+)_\d+_(\d+)$")
def tenant(k):
    k = str(k)
    if a.t > 0:
        m = R3.match(k)
        if m: return f"{m[1]}_op{m[2]}_t{int(m[3]) % a.t}"
        m = R46.match(k)
        if m: return f"{m[1]}_op{m[2]}_t{int(m[3]) % a.t}"
    if a.group_trials > 0 and k.startswith("trial_"):
        return f"trial_ent{zlib.crc32(k.encode()) % a.group_trials}"
    return k

df = pd.read_csv(a.inp)
mp = {k: tenant(k) for k in df["api_key"].astype(str).unique()}
df["api_key"] = df["api_key"].astype(str).map(mp)
df.to_csv(a.out, index=False)
print(f"{a.out}: {len(mp)} keys -> {len(set(mp.values()))} entities")