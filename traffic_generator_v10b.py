
import random, time, json, sys, threading, requests
import traffic_generator_sybil_s4 as tg

tg.SERVER_URL = "http://127.0.0.1:8007"
STEP_AGE, STEP_INC = 5, 5000         
FRAC = 0.5                            
AGE_RNG, INC_RNG = (20, 60), (30000, 80000)   

def clip(x, lo, hi): return max(lo, min(hi, x))

def health(s, key, typ):
    h = {"X-API-Key": key, "X-Label": "Attack", "X-Type": typ}
    tg.send_req(s, f"{tg.SERVER_URL}/api/v1/health", headers=h)

def post_credit(s, key, typ, age, income):
    h = {"X-API-Key": key, "X-Label": "Attack", "X-Type": typ, "Content-Type": "application/json"}
    p = json.dumps({"age": int(age), "income": int(income), "note": tg.get_padding()})
    tg.send_req(s, f"{tg.SERVER_URL}/api/v1/credit_score", "POST", p, h)

def s5_loop(key):     
    s = requests.Session()
    while not tg.STOP_EVENT.is_set():
        if random.random() < 0.7:
            health(s, key, "S5_DistMatch"); time.sleep(random.uniform(0.5, 2.0))
        if random.random() < 0.3:
            post_credit(s, key, "S5_DistMatch", random.randint(20, 60), random.randint(30000, 80000))
        time.sleep(random.uniform(1.0, 5.0))

def s9_loop(key):      
    s = requests.Session()
    while not tg.STOP_EVENT.is_set():
        if random.random() < 0.7:
            health(s, key, "S9_StratJitter"); time.sleep(random.uniform(0.5, 2.0))
        if random.random() < 0.3:
            a, i = tg.get_grid_point(s)
            a = clip(round(a + random.uniform(-FRAC * STEP_AGE, FRAC * STEP_AGE)), *AGE_RNG)
            i = clip(round(i + random.uniform(-FRAC * STEP_INC, FRAC * STEP_INC)), *INC_RNG)
            post_credit(s, key, "S9_StratJitter", a, i)
        time.sleep(random.uniform(1.0, 5.0))

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
    for i in range(10): mk(s9_loop, f"atk_s9_{i}")
    for t in T: t.start()
    print("v10b running ")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        tg.STOP_EVENT.set(); sys.exit(0)