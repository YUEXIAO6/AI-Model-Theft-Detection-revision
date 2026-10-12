# mit_state_machine_v2.py
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class MitigationConfigV2:
    throttle_after_alerts: int = 1
    review_after_alerts: int = 3
    alert_window_s: float = 1800.0
    throttle_admit_rate: float = 0.2
    review_admit_rate: float = 0.05        # v1 是 0.0（全封）；v2 = step-up 验证 + 工单
    clean_windows_to_recover: int = 2
    review_release_clean_windows: Optional[int] = None   # None = 等待人工处理，不自动释放

def simulate_entity_policy_v2(window_starts: List[float], window_preds: List[int],
                              cfg: MitigationConfigV2) -> List[Tuple[float, str, float]]:
    out, alert_history, state, clean = [], [], "normal", 0
    for t, pred in zip(window_starts, window_preds):
        if pred == 1: alert_history.append(t); clean = 0
        else: clean += 1
        alert_history = [a for a in alert_history if t - a <= cfg.alert_window_s]
        n = len(alert_history)
        if state == "review_queue":
            if cfg.review_release_clean_windows is not None and clean >= cfg.review_release_clean_windows:
                state = "normal"; alert_history = []
            else:
                out.append((t, "review_queue", cfg.review_admit_rate)); continue
        if n >= cfg.review_after_alerts: state = "review_queue"
        elif n >= cfg.throttle_after_alerts: state = "throttled"
        elif state == "throttled" and clean >= cfg.clean_windows_to_recover: state = "normal"
        admit = {"normal": 1.0, "throttled": cfg.throttle_admit_rate, "review_queue": cfg.review_admit_rate}[state]
        out.append((t, state, admit))
    return out