# hf_collect_traces.py -- collect steady/burst traces from a live HF text-classification endpoint.
# Token is read from the environment variable HF_TOKEN; it is never written to disk.
import os, sys, json, time, argparse
import numpy as np, pandas as pd, requests

DEFAULT_URL = ("https://router.huggingface.co/hf-inference/models/"
               "distilbert/distilbert-base-uncased-finetuned-sst-2-english")
ALPHABET = list("abcdefghijklmnopqrstuvwxyz ")

def make_body(rng, mu=90, sd=28):
    n = int(np.clip(round(rng.normal(mu, sd)), 10, 400))
    text = "".join(rng.choice(ALPHABET, n)).strip() or "ok"
    return json.dumps({"inputs": text}).encode("utf-8")

def one_request(sess, url, headers, body):
    t_send = time.time(); t0 = time.perf_counter()
    try:
        r = sess.post(url, data=body, headers=headers, timeout=30)
        status, txt = r.status_code, r.text[:200]
    except requests.RequestException as e:
        status, txt = 0, str(e)[:200]
    return {"t_send": t_send, "payload_bytes": len(body),
            "latency_s": time.perf_counter() - t0, "status": status}, txt

def warmup(sess, url, headers, rng, tries=10):
    for _ in range(tries):
        rec, txt = one_request(sess, url, headers, make_body(rng))
        if rec["status"] == 200:
            print("[warmup] ok"); return
        print(f"[warmup] status={rec['status']} {txt}")
        if rec["status"] in (401, 402, 403, 404, 410):
            sys.exit("Endpoint/token problem: check the URL (--url) and the token permissions.")
        time.sleep(5)
    sys.exit("Warm-up failed.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--n_steady", type=int, default=100)
    ap.add_argument("--n_burst", type=int, default=60)
    ap.add_argument("--steady_mean", type=float, default=2.25)   # set from hf_sim_reference.py
    ap.add_argument("--out_dir", default="data/hf_traces")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tok = os.environ.get("HF_TOKEN")
    if not tok: sys.exit("Set HF_TOKEN first (set HF_TOKEN=... in cmd).")
    headers = {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    sess = requests.Session()
    warmup(sess, a.url, headers, rng)

    # steady: exponential start-to-start intervals
    rows = []; sched = time.perf_counter()
    for i in range(a.n_steady):
        d = sched - time.perf_counter()
        if d > 0: time.sleep(d)
        rec, _ = one_request(sess, a.url, headers, make_body(rng)); rec["idx"] = i; rows.append(rec)
        sched += rng.exponential(a.steady_mean)
        if i % 20 == 0: print(f"[steady] {i}/{a.n_steady}")
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "hf_steady.csv"), index=False)

    time.sleep(10)

    # burst: sleep U(0.01, 0.22) after each response
    rows = []
    for i in range(a.n_burst):
        rec, _ = one_request(sess, a.url, headers, make_body(rng)); rec["idx"] = i; rows.append(rec)
        time.sleep(rng.uniform(0.01, 0.22))
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "hf_burst.csv"), index=False)

    meta = {"url": a.url, "date": time.strftime("%Y-%m-%d %H:%M:%S"), "steady_mean": a.steady_mean,
            "n_steady": a.n_steady, "n_burst": a.n_burst, "seed": a.seed}
    json.dump(meta, open(os.path.join(a.out_dir, "hf_meta.json"), "w"), indent=2)
    for f in ("hf_steady.csv", "hf_burst.csv"):
        d = pd.read_csv(os.path.join(a.out_dir, f))
        print(f"{f}: n={len(d)} ok={int((d.status==200).sum())} "
              f"mean latency={d.loc[d.status==200,'latency_s'].mean():.3f}s")

if __name__ == "__main__":
    main()