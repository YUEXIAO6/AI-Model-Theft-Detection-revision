# common_v11.py —— 在 common_v10 基础上增加 S9；输出到 results_v11/
# 必须在 `from common_v8 import *` 之前 import
import os
import common_v10                      # 已加入 S7/S8
import common_v8 as c8

if "atk_s9" not in c8.KINDS:
    c8.KINDS = c8.KINDS + ("atk_s9",)
c8.PATTERNS_ALL.update({"S9": "S9_"})
c8.OUT_DIR = "results_v11"
os.makedirs(c8.OUT_DIR, exist_ok=True)
OUT_DIR = c8.OUT_DIR