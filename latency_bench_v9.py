
import time, collections, platform, os
import numpy as np, torch, torch.nn as nn
from spp_experiments_v5 import (SPPConfig, ModelConfig, train_rf, select_feature_columns,
    build_spp_context_features, add_window_id, compute_window_features,
    _compute_bins_from_train, _mode_from_binned, _segment_indices)
from credit_data_v8 import prepare_credit
from common_v8 import stratified_entity_split, save_json

torch.set_num_threads(1)
K, LEVELS, WARM = 20, (1, 2), 20
BATCHES = [1, 10, 100, 1000]

def stats(ts):
    a = np.asarray(ts); return {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)), "n": len(a)}

def per_window_ms(fn, X, bs, reps=5):
    
    xb = X[:bs]; fn(xb)                      # warmup
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(xb); t.append((time.perf_counter() - t0) * 1e3 / bs)
    return float(np.median(t))

def main():
    df_raw, w, kind_map, atk, cfg, win_cfg = prepare_credit()
    out = {"machine": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
           "torch_threads": 1, "rf_n_jobs": 1}
    rng = np.random.RandomState(0)

   
    df2 = add_window_id(df_raw, win_cfg, cfg)
    groups = list(df2.groupby(["api_key", "win_id"]))
    sel = rng.choice(len(groups), size=min(300, len(groups)), replace=False)
    for i in sel[:WARM]: compute_window_features(groups[i][1], cfg, win_cfg.window_s, True, True)
    t_feat = []
    for i in sel:
        g = groups[i][1]; t0 = time.perf_counter()
        compute_window_features(g, cfg, win_cfg.window_s, True, True)
        t_feat.append((time.perf_counter() - t0) * 1e3)
    out["feature_extraction_per_window"] = stats(t_feat)

 
    cols = select_feature_columns(w, "ours_all"); Xb = w[cols].to_numpy(float)
    trk, vak, tek = stratified_entity_split(kind_map, 42)
    tr = w.index[w["api_key"].isin(trk)].to_numpy()
    edges = _compute_bins_from_train(Xb[tr], 10, "quantile")
    buf = collections.deque(maxlen=K); t_spp = []
    for n, i in enumerate(rng.choice(len(w), 2000)):
        buf.append(Xb[i]); t0 = time.perf_counter()
        Xc = np.asarray(buf); parts = []
        for lv in LEVELS:
            for s, e in _segment_indices(len(Xc), lv):
                seg = Xc[s:e]
                if seg.size == 0: parts.append(np.zeros(3 * Xc.shape[1])); continue
                mode = np.array([_mode_from_binned(seg[:, j], edges[j]) for j in range(Xc.shape[1])])
                parts.append(np.concatenate([seg.mean(0), seg.max(0), mode]))
        np.concatenate(parts)
        if n >= WARM: t_spp.append((time.perf_counter() - t0) * 1e3)
    out["spp_incremental_update"] = stats(t_spp)

   
    Xs, _ = build_spp_context_features(w, cols, "api_key", "win_start", SPPConfig(True, LEVELS, K, "past"), tr)
    X = np.hstack([Xb, Xs]); y = w["y_binary"].to_numpy(int)
    clf = train_rf(X[tr], y[tr], ModelConfig(class_weight="balanced_subsample", random_state=42, n_jobs=1))
    Xte = X[rng.choice(len(X), 1000, replace=False)]
    rf_fn = lambda xb: clf.predict_proba(xb)
    out["rf_n_jobs1_per_window_ms"] = {str(b): per_window_ms(rf_fn, Xte, b) for b in BATCHES}
    t_rf1 = []
    for i in range(WARM): clf.predict_proba(Xte[i:i+1])
    for i in range(500):
        t0 = time.perf_counter(); clf.predict_proba(Xte[i % 1000:i % 1000 + 1]); t_rf1.append((time.perf_counter() - t0) * 1e3)
    out["rf_inference_single_njobs1"] = stats(t_rf1)
    clf.set_params(n_jobs=-1)
    out["rf_n_jobs_all_per_window_ms"] = {str(b): per_window_ms(rf_fn, Xte, b) for b in BATCHES}
    clf.set_params(n_jobs=1)

   
    class G(nn.Module):
        def __init__(s, d=50, h=32):
            super().__init__(); s.gru = nn.GRU(d, h, batch_first=True); s.att = nn.Linear(h, 1); s.fc = nn.Linear(h, 1)
        def forward(s, x):
            o, _ = s.gru(x); a = torch.softmax(s.att(o), 1); return s.fc((a * o).sum(1)).squeeze(-1)
    g = G().eval(); xall = torch.randn(1000, 10, 50)
    with torch.no_grad():
        gru_fn = lambda xb: g(xb)
        out["gru_per_window_ms"] = {str(b): per_window_ms(gru_fn, xall, b) for b in BATCHES}

   
    fe, sp = out["feature_extraction_per_window"]["median_ms"], out["spp_incremental_update"]["median_ms"]
    print("\n== 各阶段（单核，中位数 ms）==")
    print(f"特征提取/窗口   : {fe:.3f}\nSPP 增量更新     : {sp:.3f}")
    print(f"RF 单次(n_jobs=1): {out['rf_inference_single_njobs1']['median_ms']:.3f}")
    print("\n== 每窗口摊销推理时间 (ms) ==")
    print(f"{'batch':>6} | {'RF n_jobs=1':>12} | {'RF n_jobs=-1':>13} | {'GRU 1线程':>10}")
    for b in BATCHES:
        print(f"{b:>6} | {out['rf_n_jobs1_per_window_ms'][str(b)]:>12.3f} | "
              f"{out['rf_n_jobs_all_per_window_ms'][str(b)]:>13.3f} | {out['gru_per_window_ms'][str(b)]:>10.3f}")
    for b in (1, 100):
        tot = fe + sp + out["rf_n_jobs1_per_window_ms"][str(b)]
        print(f"\n端到端(特征+SPP+RF, RF batch={b}): {tot:.2f} ms/窗口 -> 60s 窗口下单核约支撑 {60000/tot:.0f} 个并发 key")
    save_json("latency_v9.json", out)

if __name__ == "__main__":
    main()