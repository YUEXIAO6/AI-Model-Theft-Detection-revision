
from typing import List, Tuple
import math
import numpy as np
import pandas as pd

from spp_experiments_v5 import (
    ExperimentConfig, WindowConfig, SPPConfig, ModelConfig,
    load_and_prepare_df, build_windows_df, train_rf,
    select_feature_columns, _compute_bins_from_train, _mode_from_binned,
)

def _time_segment_indices(timestamps: np.ndarray, level: int) -> List[Tuple[int, int]]:
 
    n = len(timestamps)
    segs = 2 ** (level - 1)
    if n <= 0:
        return [(0, 0)]
    t_min, t_max = float(timestamps[0]), float(timestamps[-1])
    if t_max <= t_min:
       
        return [(0, n)] * segs if segs == 1 else [(0, n)] + [(n, n)] * (segs - 1)

    bounds = []
    edges = np.linspace(t_min, t_max, segs + 1)
    for i in range(segs):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            idx = np.where((timestamps >= lo) & (timestamps <= hi))[0]
        else:
            idx = np.where((timestamps > lo) & (timestamps <= hi))[0]
        if len(idx) == 0:
            bounds.append((0, 0))
        else:
            bounds.append((int(idx[0]), int(idx[-1]) + 1))
    return bounds

def build_spp_context_features_timebased(windows_df, feature_cols, group_col, time_col, spp_cfg, train_idx_for_bins):
    df = windows_df.copy()
    df["_row_id"] = np.arange(len(df))
    df = df.sort_values([group_col, time_col, "_row_id"])
    X_train_for_bins = windows_df.loc[train_idx_for_bins, feature_cols].to_numpy(dtype=float)
    bin_edges = _compute_bins_from_train(X_train_for_bins, n_bins=spp_cfg.mode_bins, strategy=spp_cfg.mode_strategy)
    d = len(feature_cols)
    n_segments_total = sum(2 ** (lv - 1) for lv in spp_cfg.levels)
    ctx = np.zeros((len(df), n_segments_total * 3 * d), dtype=float)

    for _, g in df.groupby(group_col, sort=False):
        idx = g.index.to_numpy()
        Xg = g[feature_cols].to_numpy(dtype=float)
        Tg = g[time_col].to_numpy(dtype=float)
        for local_t in range(len(g)):
            if spp_cfg.context == "full":
                Xctx, Tctx = Xg, Tg
            else:
                end_t = local_t + 1
                lb = int(spp_cfg.lookback_windows) if spp_cfg.lookback_windows is not None else 0
                beg_t = max(0, end_t - lb) if lb > 0 else 0
                Xctx, Tctx = Xg[beg_t:end_t, :], Tg[beg_t:end_t]
            parts = []
            for lv in spp_cfg.levels:
                for (s, e) in _time_segment_indices(Tctx, lv):   
                    segX = Xctx[s:e, :]
                    if segX.size == 0:
                        parts.append(np.zeros((3 * d,), dtype=float)); continue
                    mean_v, max_v = np.mean(segX, axis=0), np.max(segX, axis=0)
                    mode_v = np.zeros((d,), dtype=float)
                    for j in range(d):
                        mode_v[j] = _mode_from_binned(segX[:, j], bin_edges[j])
                    parts.append(np.concatenate([mean_v, max_v, mode_v], axis=0))
            ctx[int(df.loc[idx[local_t], "_row_id"]), :] = np.concatenate(parts, axis=0)
    return ctx