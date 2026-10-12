# traffic_generator_v7_2.py
import zlib, random, time, json, sys, threading, requests
import traffic_generator_sybil_s4 as tg

tg.SERVER_URL = "http://127.0.0.1:8005"
RUN = time.strftime("%m%d%H%M%S")           


N_POOL, N_SYBIL = 40, 12
POOL = [f"10.200.0.{i}" for i in range(1, N_POOL + 1)]
SYBIL_POOL = POOL[::3][:N_SYBIL]             

_benign = ([f"norm_{i}" for i in range(80)] + [f"batch_{i}" for i in range(10)]
           + [f"monitor_{i}" for i in range(10)])
random.Random(7).shuffle(_benign)            
BENIGN_IP = {k: POOL[j % N_POOL] for j, k in enumerate(_benign)}   

def _h(k): return zlib.crc32(k.encode())

def ip_for(key):                             
    if key in BENIGN_IP: return BENIGN_IP[key]
    if key.startswith(("atk_s4_sybil", "atk_s6_hybrid")):
        return SYBIL_POOL[_h(key) % N_SYBIL]
    return POOL[_h(key) % N_POOL]            

_orig = tg.send_req
def send_req(session, url, method="GET", data=None, headers=None):
    headers = dict(headers or {})
    headers["X-Fake-IP"] = ip_for(headers["X-API-Key"])
    _orig(session, url, method, data, headers)
tg.send_req = send_req

HEALTH = lambda: f"{tg.SERVER_URL}/api/v1/health"
CREDIT = lambda: f"{tg.SERVER_URL}/api/v1/credit_score"
_cnt = [0]; _lock = threading.Lock()
def _next(prefix):
    with _lock:
        _cnt[0] += 1
        return f"{prefix}_{RUN}_{_cnt[0]}"

def _post_credit(s, h, age, income):
    p = json.dumps({"age": age, "income": income, "note": tg.get_padding()})
    tg.send_req(s, CREDIT(), "POST", p, dict(h, **{"Content-Type": "application/json"}))


def s5_loop(key):
    s = requests.Session()
    h = {"X-API-Key": key, "X-Label": "Attack", "X-Type": "S5_DistMatch"}
    while not tg.STOP_EVENT.is_set():
        if random.random() < 0.7:
            tg.send_req(s, HEALTH(), headers=h); time.sleep(random.uniform(0.5, 2.0))
        if random.random() < 0.3:
            _post_credit(s, h, random.randint(20, 60), random.randint(30000, 80000))
        time.sleep(random.uniform(1.0, 5.0))


def s4_orch():
    s = requests.Session()
    while not tg.STOP_EVENT.is_set():
        key = _next("atk_s4_sybil")
        h = {"X-API-Key": key, "X-Label": "Attack", "X-Type": "S4_Sybil"}
        if random.random() < 0.5:
            tg.send_req(s, HEALTH(), headers=h); time.sleep(random.uniform(0.5, 2.0))
        for _ in range(random.randint(1, 2)):
            a, i = tg.get_grid_point(s); _post_credit(s, h, a, i)
            time.sleep(random.uniform(1.0, 5.0))
        time.sleep(random.uniform(3.0, 10.0))

def s6_orch():
    s = requests.Session()
    while not tg.STOP_EVENT.is_set():
        key = _next("atk_s6_hybrid")
        h = {"X-API-Key": key, "X-Label": "Attack", "X-Type": "S6_Hybrid"}
        if random.random() < 0.7:
            tg.send_req(s, HEALTH(), headers=h); time.sleep(random.uniform(0.5, 2.0))
        a, i = tg.get_grid_point(s); _post_credit(s, h, a, i)
        time.sleep(random.uniform(3.0, 10.0))

def trial_orch():                             
    s = requests.Session()
    while not tg.STOP_EVENT.is_set():
        key = _next("trial")
        h = {"X-API-Key": key, "X-Label": "Normal", "X-Type": "Trial"}
        if random.random() < 0.6:
            tg.send_req(s, HEALTH(), headers=h); time.sleep(random.uniform(0.5, 2.0))
        for _ in range(random.randint(1, 2)):
            _post_credit(s, h, random.randint(20, 60), random.randint(30000, 80000))
            time.sleep(random.uniform(1.0, 5.0))
        time.sleep(random.uniform(3.0, 10.0))

if __name__ == "__main__":
    T = []
    mk = lambda f, *a: T.append(threading.Thread(target=f, args=a, daemon=True))
    for i in range(80): mk(tg.user_lifecycle, "Normal", f"norm_{i}")
    for i in range(10): mk(tg.user_lifecycle, "Batch", f"batch_{i}")
    for i in range(10): mk(tg.user_lifecycle, "Monitor", f"monitor_{i}")
    for i in range(10): mk(tg.user_lifecycle, "S0", f"atk_s0_{i}")
    for i in range(10): mk(tg.user_lifecycle, "S1", f"atk_s1_{i}")
    for i in range(10): mk(tg.user_lifecycle, "S2", f"atk_s2_{i}")
    for _ in range(5):  mk(tg.user_s3_rotation)
    for i in range(10): mk(s5_loop, f"atk_s5_{i}")
    for _ in range(4):  mk(s4_orch)
    for _ in range(4):  mk(s6_orch)
    for _ in range(8):  mk(trial_orch)
    for t in T: t.start()
    print(f"v7_2 running (RUN={RUN})")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        tg.STOP_EVENT.set(); sys.exit(0)