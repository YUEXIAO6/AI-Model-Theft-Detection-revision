
import os
import common_v8 as c8

for k in ("atk_s7", "atk_s8"):
    if k not in c8.KINDS:
        c8.KINDS = c8.KINDS + (k,)
c8.PATTERNS_ALL.update({"S7": "S7_", "S8": "S8_"})
c8.OUT_DIR = "results_v10"
os.makedirs(c8.OUT_DIR, exist_ok=True)
OUT_DIR = c8.OUT_DIR