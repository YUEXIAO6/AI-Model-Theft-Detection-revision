# Detecting Stealthy AI Model Extraction via Multi-Scale Traffic Analysis

Code, simulators, logs and result files for the manuscript
**"A Management Framework for AI Service Security: Detecting Stealthy Model Extraction via Multi-Scale Traffic Analysis"** (Y. Chen, B. Xiao), under revision at IEEE TNSM.

The logs are produced by the simulators in this repository; they contain no real customer data.
Code for the first-round submission is kept at `AI-Model-Theft-Detection-v1`.

## What is here
- **Simulators**: `server_*.py` (API gateway with logging middleware) and `traffic_generator_*.py` (benign clients and attack strategies S0-S6, S9).
- **Feature pipeline**: `spp_experiments_v5*.py` (50-D window features, Spatial Pyramid Pooling), `window_relabel_credit_only.py` (credit-only window labels), `common_v8.py` (hashed keys, stratified entity-disjoint split, 1% normal-FPR calibration).
- **Evaluation scripts**: one script per table or section (map below).
- **Results**: `results_v8/`, `results_v10/`, `results_v11/` (JSON, mean/std over partitions).

## Setup
```
pip install -r requirements.txt
```
Run all commands from the repository root. Logs go in the root directory (scripts read `traffic_logs*.csv` from the working directory).

| Environment | Python | numpy | pandas | scikit-learn | scipy | torch | xgboost | CPU |
|---|---|---|---|---|---|---|---|---|
| Authors' workstation (all paper results) | 3.10 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Google Colab (reproduction check, Oct 2026) | 3.13 | 2.1.3 | 2.2.3 | 1.6.1 | 1.16.3 | 2.11.0 (CPU) | 3.4.1 | Intel Xeon @ 2.20 GHz, 2 vCPU |

Reproduction check on Colab (clean environment): the PRADA baseline reproduced exactly; RF+SPP for partition 42 within 1 point; single-window RF within 4 points; per-window latency 36.6 ms vs. 33.6 ms. Different library versions can move single-partition numbers by a few points; the 10-partition means in the paper are the reference. Seeds are 42-51.

## Data
`data/traffic_logs.csv` is the main log (275,982 requests). Larger logs are archived at **TODO: Zenodo DOI** and must be placed in the repository root:
`traffic_logs_v10b.csv` (S5/S9), `traffic_logs_v12.csv` (tenant experiment), `traffic_logs_v7_2.csv` (IP-level), `traffic_logs_drift.csv`, `traffic_logs_insurance.csv`.

Regenerating a log: start the matching `server_*.py` in one terminal and the matching `traffic_generator_*.py` in another, run about 2 hours, stop with Ctrl+C. New logs give slightly different numbers.

## Paper item to command

| Paper item | Command | Output |
|---|---|---|
| Table VII (RF, XGBoost, L7-only, time-based) | `python eval_unified_v8.py --seeds 10` | `results_v8/unified_v8.json` |
| Table VII (GRU) | `python dl_baselines_v8_multiseed.py --seeds 10` | `results_v8/gru_multiseed_v8.json` |
| Table VII (single window) | `python eval_singlewindow_v9.py --seeds 10` | `results_v8/singlewindow_v9.json` |
| Table VI (GRU grid search) | `python dl_baselines_v6_gridsearch.py` | stdout |
| Table V (HuggingFace fidelity) | TODO scripts | TODO |
| Table VIII (drift) | `server_drift.py`, `traffic_generator_drift_v11.py`, `python robustness_eval_drift_v2.py` | stdout |
| Fig. 4 (sensitivity) | `python sensitivity_analysis.py` | `sensitivity_w_k.png` |
| Table IX (insurance API) | `cd server_insurance && python eval_insurance_v8.py --seeds 10` | `results_v8/insurance_v8.json` |
| Table X (response policy, d=0) | `python mit_eval_v8.py --seeds 10` | `results_v8/mitigation_v8.json` |
| Table X (d=0,1,2) | `python mit_eval_v10_delay.py --seeds 10` | `results_v10/mitigation_v10_delay.json` |
| Benign cost by client kind | `python mit_benign_by_kind_v8.py --seeds 10` | `results_v8/mit_benign_by_kind_v8.json` |
| Sec. VI-D (latency) | `python latency_bench_v9.py` | `results_v8/latency_v9.json` |
| Table XI (tenants) | `python make_tenant_log_v12.py --t 2 --out traffic_logs_v12_t2.csv`, then `python eval_tenant_v12.py --seeds 10 --csvs traffic_logs_v12_key.csv traffic_logs_v12_t2.csv ...`; trial-grouping rows: `python eval_tenant_v13.py ...` | `results_v8/tenant_*.json` |
| Sec. VI-F (PRADA) | `python prada_baseline_v8.py --seeds 10`, `python prada_relaxed_v8.py --seeds 10` | `results_v8/prada_*.json` |
| Sec. VI-G (cold start, running example) | `python cold_start_v13.py --seeds 5`, `python running_example_v13.py` | `results_v8/cold_start_v13.json` |
| Sec. VI-B scope (S5/S9, leave-strategy-out) | `python eval_s9_v11.py --csv traffic_logs_v10b.csv --seeds 10 --holdout atk_s5,atk_s9`; `python eval_loso_v11.py --seeds 5`; `python s9_extraction_value_v11.py` | `results_v11/*.json` |
| Sec. VI-A.2 (no server features, +0.11 s) | `python eval_noserver_v12.py --seeds 10` | `results_v8/noserver_v12.json` |
| Sec. IV-B.2 (mutual information) | `python spp_information_gain.py` | stdout |
| Request-level attack share | `python attack_share_v9.py` | stdout |
| IP-level aggregation (reported as unusable) | `python eval_granularity_v7_3.py --csv traffic_logs_v7_2.csv --seeds 10` | `results_v8/granularity_v7_3.json` |

Approximate run times on the authors' workstation: `eval_unified_v8` about 1 h, GRU about 30 min, single-window about 20 min, PRADA about 15 min each, `eval_s9_v11` about 3 h per setting, `eval_loso_v11` about 1 h.

## Scope and limitations
All traffic is simulated. S1 detection depends on parameter-traversal features that exploit the regular query grid of the simulated attackers; queries drawn from the benign distribution (S5) or from a jittered grid (S9) are essentially not detected. See Sec. VI-B and VII of the paper.

## License and contact
MIT License (see `LICENSE`). Contact: yuelin.chen@connect.polyu.hk
