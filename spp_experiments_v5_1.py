# spp_experiments_v5_1.py
# 不修改原始 spp_experiments_v5.py，只在此基础上扩展一个新的特征集选项 "l7_only"
from typing import List
import pandas as pd

from spp_experiments_v5 import (
    ExperimentConfig, WindowConfig, SPPConfig, ModelConfig,
    load_and_prepare_df, build_windows_df, train_rf,
    build_spp_context_features,
    select_feature_columns as _select_feature_columns_base,
)

# 与原文件 ours_extra 完全一致的 27 维 L7 语义特征列表
# （如果原文件以后改了这个列表，记得同步这里）
_OURS_EXTRA = [
    "req_cnt", "proc_dur_mean", "proc_dur_std", "iat_mean", "iat_std", "iat_min", "iat_max",
    "endpoint_nunique", "endpoint_entropy", "top_endpoint_frac", "method_nunique", "method_entropy",
    "status_2xx_frac", "status_4xx_frac", "status_5xx_frac", "bytes_per_s",
    "payload_size_entropy_fwd", "payload_size_bins_nunique_fwd",
    "payload_size_entropy_bwd", "payload_size_bins_nunique_bwd",
    "credit_unique_age", "credit_unique_income", "credit_unique_pairs",
    "credit_age_entropy", "credit_income_entropy", "credit_age_step_mean", "credit_income_step_mean",
]

def select_feature_columns(windows_df: pd.DataFrame, feature_set: str) -> List[str]:
    """
    包装原始 select_feature_columns，新增一个 "l7_only" 选项：
    完全剔除 23-D NetFlow zero-padded 模板，只保留 27-D L7 语义特征。
    其他所有 feature_set 值（baseline23 / ours_all / ours_no_payload / ours_no_semantic / iat_rate_only）
    行为与原函数完全一致，直接透传。
    """
    if feature_set == "l7_only":
        return [c for c in _OURS_EXTRA if c in windows_df.columns]
    return _select_feature_columns_base(windows_df, feature_set)