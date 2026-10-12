
from spp_experiments_v5 import ExperimentConfig, WindowConfig, load_and_prepare_df, build_windows_df
from window_relabel_credit_only import relabel_windows_credit_only
from common_v8 import hash_keys

def prepare_credit(csv="traffic_logs.csv", window_s=60, endpoint="/api/v1/credit_score"):
    cfg = ExperimentConfig(csv_file=csv, group_col="api_key")
    win_cfg = WindowConfig(window_s=window_s, attack_ratio_thr=0.1)
    df_raw = load_and_prepare_df(cfg)
    df_raw, kind_map, atk = hash_keys(df_raw)
    w = build_windows_df(df_raw, cfg, win_cfg, True, True)
    w = relabel_windows_credit_only(df_raw=df_raw, windows_df=w, window_s=window_s,
                                    group_col="api_key", credit_endpoint=endpoint)
    w = w.sort_values("win_start").reset_index(drop=True)
    w["is_atk"] = w["api_key"].isin(atk)
    return df_raw, w, kind_map, atk, cfg, win_cfg