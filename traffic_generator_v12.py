
import random, time, json, sys, threading, requests
import traffic_generator_sybil_s4 as tg

tg.SERVER_URL = "http://127.0.0.1:8008"
RUN = time.strftime("%m%d%H%M%S")      
N_OP = 10
_cnt = [0]; _lock = threading.Lock()

def _next():
    with _lock:
        _cnt[0] += 1
        return _cnt[0]

def health(s, h):  tg.send_req(s, f"{tg.SERVER_URL}/api/v1/health", headers=h)
def credit(s, h, a, i):
    p = json.dumps({"age": a, "income": i, "note": tg.get_padding()})
    tg.send_req(s, f"{tg.SERVER_URL}/api/v1/credit_score", "POST", p,
                dict(h, **{"Content-Type": "application/json"}))

def s3_op(j):                         
    s = requests.Session(); keys = [f"atk_s3_op{j}_k{i}" for i in range(10)]
    while not tg.STOP_EVENT.is_set():
        k = random.choice(keys)
        if random.random() < 0.95:
            tg.behavior_normal(k, s)
        else:
            a, i = tg.get_grid_point(s)
            credit(s, {"X-API-Key": k, "X-Label": "Attack", "X-Type": "S3_KeyRotation"}, a, i)
            time.sleep(random.uniform(1.0, 5.0))

def s4_op(j):                          
    s = requests.Session(); n = 0
    while not tg.STOP_EVENT.is_set():
        n += 1; k = f"atk_s4_op{j}_{RUN}_{n}"
        h = {"X-API-Key": k, "X-Label": "Attack", "X-Type": "S4_Sybil"}
        if random.random() < 0.5:
            health(s, h); time.sleep(random.uniform(0.5, 2.0))
        for _ in range(random.randint(1, 2)):
            a, i = tg.get_grid_point(s); credit(s, h, a, i)
            time.sleep(random.uniform(1.0, 5.0))
        time.sleep(random.uniform(3.0, 10.0))

def s6_op(j):                           
    s = requests.Session(); n = 0
    while not tg.STOP_EVENT.is_set():
        n += 1; k = f"atk_s6_op{j}_{RUN}_{n}"
        h = {"X-API-Key": k, "X-Label": "Attack", "X-Type": "S6_Hybrid"}
        if random.random() < 0.7:
            health(s, h); time.sleep(random.uniform(0.5, 2.0))
        a, i = tg.get_grid_point(s); credit(s, h, a, i)
        time.sleep(random.uniform(3.0, 10.0))

def trial_orch():                      
    s = requests.Session()
    while not tg.STOP_EVENT.is_set():
        k = f"trial_{RUN}_{_next()}"
        h = {"X-API-Key": k, "X-Label": "Normal", "X-Type": "Trial"}
        if random.random() < 0.6:
            health(s, h); time.sleep(random.uniform(0.5, 2.0))
        for _ in range(random.randint(1, 2)):
            credit(s, h, random.randint(20, 60), random.randint(30000, 80000))
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
    for j in range(N_OP): mk(s3_op, j)
    for j in range(N_OP): mk(s4_op, j)
    for j in range(N_OP): mk(s6_op, j)
    for _ in range(8):    mk(trial_orch)
    for t in T: t.start()
    print(f"v12 running (RUN={RUN})")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        tg.STOP_EVENT.set(); sys.exit(0)