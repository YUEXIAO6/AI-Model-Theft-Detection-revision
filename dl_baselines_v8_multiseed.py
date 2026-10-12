# dl_baselines_v8_multiseed.py
# [v8-fix] COLS 增加 EntFPR3
import argparse
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from spp_experiments_v5 import select_feature_columns
from credit_data_v8 import prepare_credit
from common_v8 import *

HIDDEN, LOOKBACK, EPOCHS = 32, 10, 15
COLS = list(PATTERNS_ALL) + ["Macro", "WinFPR", "EntFPR", "EntFPR3", "AlertHr", "PRAUC"]

class GRUAttention(nn.Module):
    def __init__(self, d, h=HIDDEN):
        super().__init__()
        self.gru = nn.GRU(d, h, batch_first=True); self.att = nn.Linear(h, 1); self.fc = nn.Linear(h, 1)
    def forward(self, x):
        o, _ = self.gru(x); a = torch.softmax(self.att(o), dim=1)
        return self.fc((a * o).sum(1)).squeeze(-1)

def make_seq(X, w, lookback):
    N, D = X.shape; S = np.zeros((N, lookback, D), dtype=np.float32)
    for _, g in w.groupby("api_key"):
        idx = g.index.to_numpy(); Xg = X[idx]
        for t in range(len(g)):
            seq = Xg[max(0, t - lookback + 1): t + 1]
            if len(seq) < lookback: seq = np.vstack([np.zeros((lookback - len(seq), D)), seq])
            S[idx[t]] = seq
    return S

def run(w, feat, tr, va, te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    cols = select_feature_columns(w, feat)
    X = w[cols].to_numpy(float); y = w["y_binary"].to_numpy(int); is_atk = w["is_atk"].to_numpy()
    sc = StandardScaler().fit(X[tr]); S = make_seq(sc.transform(X).astype(np.float32), w, LOOKBACK)
    Xtr, ytr = torch.tensor(S[tr]), torch.tensor(y[tr], dtype=torch.float32)
    m = GRUAttention(len(cols)); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    pos = max(float(ytr.sum()), 1.0); crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(ytr) - pos) / pos]))
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
    m.train()
    for _ in range(EPOCHS):
        for bx, by in dl:
            opt.zero_grad(); crit(m(bx), by).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p_va = torch.sigmoid(m(torch.tensor(S[va]))).numpy()
        p_te = torch.sigmoid(m(torch.tensor(S[te]))).numpy()
    thr = calibrate_thr(p_va, (y[va] == 0) & (~is_atk[va]))
    met = compute_metrics(w.iloc[te], (p_te > thr).astype(int), is_atk[te])
    met["PRAUC"] = float(average_precision_score(y[te], p_te))
    return met

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=10); a = ap.parse_args()
    _, w, kind_map, atk, _, _ = prepare_credit()
    res = {"GRU+Attn (Baseline23)": [], "GRU+Attn (Ours All)": []}
    for seed in SEEDS[:a.seeds]:
        trk, vak, tek = stratified_entity_split(kind_map, seed)
        idx = lambda S: w.index[w["api_key"].isin(S)].to_numpy()
        tr, va, te = idx(trk), idx(vak), idx(tek)
        res["GRU+Attn (Baseline23)"].append(run(w, "baseline23", tr, va, te, seed))
        res["GRU+Attn (Ours All)"].append(run(w, "ours_all", tr, va, te, seed))
        print(f"[seed={seed}] done")
    summary = print_table(res, COLS)
    save_json("gru_multiseed_v8.json", {"summary": summary, "per_seed": res,
                                        "config": {"hidden": HIDDEN, "lookback": LOOKBACK, "epochs": EPOCHS}})

if __name__ == "__main__":
    main()