"""Persist bounded Run B statuses and execute the non-SVD sensitivity branch."""
from __future__ import annotations
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_b as rb


def save_json(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2), encoding="utf-8")


def save_csv(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (HERE / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


test4 = {
    "test": "TEST 4 — LARGER NEIGHBOURHOOD", "status": "BLOCKED",
    "configuration": {"rho": .1, "S": 1.0, "abs_tolerance": 1e-6,
                       "relative_tolerance": 1e-5, "outer_cap": 200,
                       "inner_solver_maxiter": 300,
                       "changed_only": "neighbourhood sample count n=1000 to n=3000"},
    "record_results": [
        {"record_index": 2, "subproblem": "record_2_failed_001.npz", "status": "BLOCKED",
         "attempted": "record 2 only; n=1000 then n=3000 planned",
         "completed": "training reconstruction; n=1000 X, y, and first E-step responsibilities matched the persisted subproblem within thresholds 1e-10, 1e-10, and 1e-8 respectively",
         "exact_error": "KeyboardInterrupt: dense full-spectrum SVD did not return; manually stopped after approximately 11 minutes as too slow. No SVD result or solver metrics completed."},
        {"record_index": 4, "subproblem": "record_4_failed_001.npz", "status": "NOT RUN",
         "reason": "Record 2 did not finish within the practical time budget."}],
    "runtime_seconds_before_stop_approx": 660,
}
save_json("test4_larger_neighbourhood.json", test4)
save_csv("test4_larger_neighbourhood.csv", [
    {"record_index": 2, "subproblem": "record_2_failed_001.npz", "status": "BLOCKED",
     "exact_error": test4["record_results"][0]["exact_error"]},
    {"record_index": 4, "subproblem": "record_4_failed_001.npz", "status": "NOT RUN",
     "reason": test4["record_results"][1]["reason"]},
])

test5 = {
    "test": "TEST 5 — CONNECTED-FEATURE ABLATION", "status": "BLOCKED",
    "feature_space_note": "The 608-coordinate ablation is a changed feature space and a different optimisation problem.",
    "configuration": {"full_feature_count": 2381, "graph_touched_feature_count": 608,
                       "rho": .1, "S": 1.0, "abs_tolerance": 1e-6,
                       "relative_tolerance": 1e-5, "outer_cap": 200,
                       "inner_solver_maxiter": 300},
    "record_results": [{"status": "BLOCKED",
                        "exact_error": "KeyboardInterrupt: dense full-spectrum SVD for the first (record 2, all 2,381 coordinates) case did not return; manually stopped after 300 seconds as too slow. No Test 5 rank or solver metrics completed."},
                       {"record_index": 4, "status": "NOT RUN", "reason": "Stopped after record 2 full-spectrum SVD exceeded the practical time budget."}],
    "runtime_seconds_before_stop_approx": 300,
}
save_json("test5_ablation.json", test5)
save_csv("test5_ablation.csv", [
    {"record_index": 2, "feature_space": "all_2381_coordinates", "status": "BLOCKED",
     "exact_error": test5["record_results"][0]["exact_error"]},
    {"record_index": 4, "status": "NOT RUN", "reason": test5["record_results"][1]["reason"]},
])

started = time.perf_counter()
test6, rows = rb.test6()
save_json("test6_sensitivity.json", test6)
save_csv("test6_sensitivity.csv", rows)
runtime = {"python": sys.version, "numpy": rb.np.__version__, "scipy": rb.scipy.__version__,
           "platform": platform.platform(), "processor": platform.processor(),
           "logical_cpu_count": os.cpu_count(), "training_data_only": True,
           "official_test_records_accessed": False, "malware_executed": False,
           "run_a_edited": False, "run_b_test6_seconds": time.perf_counter() - started,
           "svd": "Run B Test 4 and Test 5 SVD branches manually stopped as recorded; no partial spectrum presented."}
save_json("run_metadata.json", runtime)

readme = f"""# Run B — bounded controlled tests

## DONE / NOT RUN / BLOCKED

- TEST 4 — BLOCKED: record 2 training inputs were reconstructed and matched the saved n=1,000 data and initial responsibilities. The first full-spectrum SVD did not return after approximately 11 minutes and was manually stopped. Record 4 was NOT RUN because record 2 did not finish.
- TEST 5 — BLOCKED: the first full-spectrum SVD for record 2 with 2,381 coordinates did not return after 300 seconds and was manually stopped. No rank or solver metrics completed; record 4 was NOT RUN.
- TEST 6 — {test6['status']}: see `test6_sensitivity.json/.csv`; any remaining cases after the 900-second branch budget are listed as NOT RUN.

## Data, controls, and methods

- Used only the two saved first-entry weighted subproblems from records 2 and 4 and the training-only data needed to reconstruct the larger neighborhood. No official test records or malware were accessed.
- Test 4 inspected `generate_raw_neighborhood`, `load_labeled_training`, the scaler/model setup, and LEMNA's initialization. It used the same source record, training median, StandardScaler, LGBM model, model/sample seed, 50% replacement policy, fusion graph, rho=0.1, S=1, tolerances 1e-6 / 1e-5, and 200 outer iterations. Only neighborhood sample count was changed. The reconstructed n=1,000 record 2 design, predictions, and first E-step responsibilities matched the persisted input within explicit numeric limits.
- Test 5, if completed, would compare all 2,381 coordinates with 608 graph-touched coordinates while holding saved row responsibilities fixed. This reduced feature space is a different optimisation problem. No metrics completed.
- Test 6 uses the exact saved Run A data, original tolerances at caps 200, 1,000 and 2,000, and one looser setting (abs=1e-5, rel=1e-4) at 1,000. It reports outer residual certificates and inner iteration-limit stops separately. SciPy stop code 7 denotes reaching the inner iteration limit; Cholesky is direct and has no iterative limit.
- Run A files were not changed. No paper, GitHub repository, or Zenodo record was changed or created.

## Runtime and libraries

- Python `{sys.version.split()[0]}`, NumPy `{rb.np.__version__}`, SciPy `{rb.scipy.__version__}`.
- Platform `{platform.platform()}`, processor `{platform.processor()}`, logical CPUs `{os.cpu_count()}`.
- See `run_metadata.json` for Test 6 runtime. Dense SVD interruption details are included in the corresponding JSON and CSV.

## Artifacts

- `test4_larger_neighbourhood.json/.csv`
- `test5_ablation.json/.csv`
- `test6_sensitivity.json/.csv`
- `run_b.py`, `run_b_test4_interrupted.py`, `run_b_test5_interrupted.py`, `finalize_run_b.py`
- `run_metadata.json`
"""
(HERE / "README_RUN_B.md").write_text(readme, encoding="utf-8")
print(json.dumps({"test6_status": test6["status"], "records": len(test6.get("results", [])),
                  "not_run_count": len(test6.get("not_run", [])),
                  "runtime_seconds": runtime["run_b_test6_seconds"]}, indent=2))
