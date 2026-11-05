# hf_fidelity_analysis.py -- KS_boot, Cohen's d, OVL: live HF traces vs simulator
import argparse, json, os
import numpy as np, pandas as pd
from scipy.stats import ks_2samp, gaussian_kde
from hf_common import load_sim_samples

TRAPZ = getattr(np, "trapezoid", None) or np.trapz

def real_samples(path):
    d = pd.read_csv(path).sort_values("t_send")
    ok = d[d["status"] == 200]
    return {"iat": np.diff(d["t_send"].to_numpy(float)),
            "payload": ok["payload_bytes"].to_numpy(float),
            "latency": ok["latency_s"].to_numpy(float)}

def ks_boot(real, sim, n_iter=1000, seed=0):
    rng = np.random.default_rng(seed); n = min(len(real), len(sim))
    v = [ks_2samp(rng.choice(real, n, replace=True),
                  rng.choice(sim, n, replace=False)).statistic for _ in range(n_iter)]
    return float(np.median(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

def cohen_d(a, b):
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return float(abs(a.mean() - b.mean()) / sp) if sp > 0 else float("nan")

def ovl(a, b):
    if a.std() == 0 or b.std() == 0: return float("nan")
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max()); pad = 0.1 * (hi - lo)
    g = np.linspace(lo - pad, hi + pad, 2000)
    return float(TRAPZ(np.minimum(gaussian_kde(a)(g), gaussian_kde(b)(g)), g))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="traffic_logs.csv")
    ap.add_argument("--trace_dir", default="data/hf_traces")
    ap.add_argument("--iat_scope", default="all", choices=["all", "credit"])
    ap.add_argument("--out", default="results_v8/hf_fidelity.json")
    a = ap.parse_args()

    sim = load_sim_samples(a.csv, a.iat_scope)
    real = {"steady": real_samples(os.path.join(a.trace_dir, "hf_steady.csv")),
            "burst": real_samples(os.path.join(a.trace_dir, "hf_burst.csv"))}
    names = {"iat": "IAT (s)", "payload": "Payload (B)", "latency": "Latency (s)"}
    res, tex = {}, []
    print(f"{'Pattern':<8}{'Feature':<13}{'Real mu':>9}{'Sim mu':>9}{'KSboot [95% CI]':>24}{'d':>7}{'OVL':>7}  n_real/n_sim  Grade")
    for pat in ("steady", "burst"):
        for f in ("iat", "payload", "latency"):
            r, s = real[pat][f], sim[pat][f]
            ks, lo, hi = ks_boot(r, s); d = cohen_d(r, s); o = ovl(r, s)
            grade = "*" if f == "latency" else ("ok" if (ks < 0.20 and d < 0.50) else "CHECK")
            print(f"{pat:<8}{names[f]:<13}{r.mean():9.3f}{s.mean():9.3f}{ks:8.3f} [{lo:.3f}, {hi:.3f}]{d:7.2f}{o:7.2f}  {len(r)}/{len(s)}  {grade}")
            res[f"{pat}|{f}"] = dict(real_mu=r.mean(), sim_mu=s.mean(), ks=ks, ci=[lo, hi], d=d, ovl=o,
                                     n_real=len(r), n_sim=len(s), grade=grade)
            g = r"$\ast$" if f == "latency" else (r"\checkmark" if grade == "ok" else "?")
            tex.append(f"{pat.capitalize()} & {names[f]} & {r.mean():.3f} & {s.mean():.3f} & "
                       f"{ks:.3f} [{lo:.3f}, {hi:.3f}] & {d:.2f} & {o:.2f} & {g} \\\\")
    print("\nLaTeX rows:\n" + "\n".join(tex))
    for pat in ("steady", "burst"):
        p = real[pat]["payload"]
        print(f"{pat}: real payload {p.mean():.1f}+-{p.std():.1f} B; sim {sim[pat]['payload'].mean():.1f}+-{sim[pat]['payload'].std():.1f} B")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"iat_scope": a.iat_scope, "results": res}, open(a.out, "w"), indent=2, default=float)
    print(f"[*] saved {a.out}")

if __name__ == "__main__":
    main()