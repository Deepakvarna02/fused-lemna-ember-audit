"""Record the bounded Test 6 interruption after the numerical run was stopped."""
from __future__ import annotations
import csv, json, os, platform, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = ROOT / "experiments" / "results" / "raw"
saved_run_a = json.loads((RAW / "LEMNA_RANK_AWARE_SOLVER_VALIDATION.json").read_text(encoding="utf-8"))
cases = [
    {"outer_cap": cap, "abs_tolerance": a, "relative_tolerance": r,
     "record_2_status": "BLOCKED — execution interrupted before result files were written",
     "record_4_status": "BLOCKED — execution interrupted before result files were written"}
    for cap, a, r in [(200, 1e-6, 1e-5), (1000, 1e-6, 1e-5),
                      (2000, 1e-6, 1e-5), (1000, 1e-5, 1e-4)]
]
exact = "KeyboardInterrupt: manually stopped Test 6 after approximately 660 seconds because no completed test result had been persisted and execution was too slow for the remaining task budget. This manual stop occurred before the script's 900-second branch cap."
result = {
    "test": "TEST 6 — ITERATION / TOLERANCE SENSITIVITY",
    "status": "BLOCKED",
    "exact_error": exact,
    "configuration": {"rho": 0.1, "S": 1.0,
                       "objective": "0.5 * sum(w*(X*beta-y)^2)/sum(w)",
                       "graph": "unchanged 605-edge EMBER fusion graph",
                       "inner_solver_maxiter": 300,
                       "inner_limit_stop_code": 7,
                       "requested_cases": cases},
    "results": [],
    "not_reported": "Caps 1,000 and 2,000, plus the looser 1,000-iteration setting, were not completed. No metrics are inferred from incomplete in-memory work.",
    "runtime_seconds_before_stop_approx": 660,
}
for subproblem in saved_run_a["subproblems"]:
    if subproblem["file"] not in ("record_2_failed_001.npz", "record_4_failed_001.npz"):
        continue
    for method, saved in subproblem["methods"].items():
        final = saved["final"]
        beta_step = saved["beta_step"]
        result["results"].append({
            "record_index": subproblem["record_index"], "subproblem": subproblem["file"],
            "outer_cap": 200, "outer_iterations": saved["iterations"],
            "abs_tolerance": 1e-6, "relative_tolerance": 1e-5,
            "method": method,
            "residual_certificate": bool(final["primal_residual"] <= final["primal_tolerance"] and final["dual_residual"] <= final["dual_tolerance"]),
            "primal_residual": final["primal_residual"], "dual_residual": final["dual_residual"],
            "primal_tolerance": final["primal_tolerance"], "dual_tolerance": final["dual_tolerance"],
            "raw_beta_tv": saved["raw_tv"], "objective": final["weighted_regression_objective"],
            "runtime_seconds": saved["runtime_seconds"],
            "inner_solver_maxiter": None if method == "cholesky_jitter" else 300,
            "final_inner_stop_code": None if method == "cholesky_jitter" else beta_step["istop"],
            "final_inner_iterations": None if method == "cholesky_jitter" else beta_step["iterations"],
            "inner_solver_all_steps_converged": saved["beta_step_converged_all"],
            "source": "reused exact cap-200 Run A persisted validation; read-only",
        })
(HERE / "test6_sensitivity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
with (HERE / "test6_sensitivity.csv").open("w", newline="", encoding="utf-8") as f:
    fields = list(dict.fromkeys(k for row in result["results"] for k in row))
    fields.extend(["record_2_status", "record_4_status", "exact_error"])
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for row in result["results"]:
        w.writerow(row)
    for row in cases:
        w.writerow({**row, "exact_error": exact})
    for row in cases:
        w.writerow({**row, "exact_error": exact})
runtime = {"python": sys.version, "numpy": "2.2.6", "scipy": "1.15.3",
           "platform": platform.platform(), "processor": platform.processor(),
           "logical_cpu_count": os.cpu_count(), "training_data_only": True,
           "official_test_records_accessed": False, "malware_executed": False,
           "run_a_edited": False, "test6_elapsed_seconds_before_stop_approx": 660,
           "svd_branch_method": "numpy.linalg.svd(compute_uv=False, full_matrices=False); interrupted branches report no partial spectrum."}
(HERE / "run_metadata.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
readme = """# Run B — bounded controlled tests

## DONE / NOT RUN / BLOCKED

- TEST 4 — BLOCKED: training reconstruction passed its n=1,000 X, prediction, and initial-responsibility checks. The full dense SVD did not return after approximately 11 minutes and was manually stopped. Record 4 was NOT RUN because record 2 did not finish.
- TEST 5 — BLOCKED: the full-spectrum SVD for record 2 with all 2,381 coordinates did not return after 300 seconds and was manually stopped. No rank or solver metrics completed; record 4 was NOT RUN.
- TEST 6 — BLOCKED: the solver branch was manually stopped after approximately 660 seconds, before its configured 900-second cap, because it had not completed the requested sensitivity set. The cap-200 / original-tolerance baseline below was reused read-only from the persisted Run A validation. The 1,000, 2,000 and looser-tolerance cases are NOT RUN.

## Data, controls, and methods

- Used only the two saved first-entry weighted subproblems from records 2 and 4 and the training-only data needed to reconstruct the larger neighborhood. No official test records or malware were accessed.
- Test 4 inspected `generate_raw_neighborhood`, `load_labeled_training`, scaler/model setup, and LEMNA initialization. It used the same source record, training median, StandardScaler, LGBM model, seed, replacement policy, fusion graph, rho=0.1, S=1, tolerances 1e-6 / 1e-5, and 200 outer iterations. Only neighborhood sample count was to change. The reconstructed n=1,000 record 2 design, predictions, and first E-step responsibilities matched the persisted input within thresholds 1e-10, 1e-10, and 1e-8.
- Test 5 was specified to compare all 2,381 coordinates with the 608 graph-touched coordinates while holding saved row responsibilities fixed. This reduced feature space is a different optimization problem. No metrics completed.
- Test 6 uses the saved Run A data, original tolerances at caps 200, 1,000 and 2,000, and abs=1e-5 / rel=1e-4 at 1,000. Exact cap-200 metrics for both persisted inputs were reused read-only from `experiments/results/raw/LEMNA_RANK_AWARE_SOLVER_VALIDATION.json`. The attempted sensitivity branch was manually stopped after approximately 660 seconds. The output keeps final inner LSMR/LSQR stop code and iteration count separate from the outer cap; Cholesky is direct.
- Run A files were not changed. No paper, GitHub repository, or Zenodo record was changed or created.

## Runtime and libraries

- Python 3.12.4, NumPy 2.2.6, SciPy 1.15.3.
- Platform, processor, and logical CPU count are in `run_metadata.json`.
- Full singular spectra were requested through dense NumPy SVD. Because SVD did not finish within the recorded practical limits, no partial spectrum or rank is claimed.

## Artifacts

- `test4_larger_neighbourhood.json/.csv`
- `test5_ablation.json/.csv`
- `test6_sensitivity.json/.csv`
- `run_b.py`, `run_b_test4_interrupted.py`, `run_b_test5_interrupted.py`, `finalize_run_b.py`, `record_test6_interruption.py`
- `run_metadata.json`
"""
(HERE / "README_RUN_B.md").write_text(readme, encoding="utf-8")
