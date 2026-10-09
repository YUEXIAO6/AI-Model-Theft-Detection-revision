# common_v10.py —— 不改 common_v8.py：扩展 S7/S8，输出重定向到 results_v10/
# 必须在 `from common_v8 import *` 之前 import 本模块
import os
import common_v8 as c8

for k in ("atk_s7", "atk_s8"):
    if k not in c8.KINDS:
        c8.KINDS = c8.KINDS + (k,)
c8.PATTERNS_ALL.update({"S7": "S7_", "S8": "S8_"})
c8.OUT_DIR = "results_v10"
os.makedirs(c8.OUT_DIR, exist_ok=True)
OUT_DIR = c8.OUT_DIR