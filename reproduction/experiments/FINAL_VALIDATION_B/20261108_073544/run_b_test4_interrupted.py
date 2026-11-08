"""Bounded controlled Run B analyses for persisted EMBER training-only inputs."""
from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import lsqr, lsmr
from scipy.linalg import cho_factor, cho_solve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPERIMENTS = ROOT / "experiments"
RAW = EXPERIMENTS / "results" / "raw"
FAILED = RAW / "failed_subproblems"
sys.path.insert(0, str(EXPERIMENTS))

from scripts.run_pilot2 import generate_raw_neighborhood, load_labeled_training, SEED
from src.protocol import ember_fusion_edges
from src.lemna import _difference_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

RHO = 0.1
S = 1.0
ABS_TOL = 1e-6
REL_TOL = 1e-5
INNER_MAXITER = 300
METHODS = ("cholesky_jitter", "lsmr", "lsqr")
BRANCH_BUDGET_SECONDS = 900.0
RECORDS = (2, 4)
INPUT_FILES = {2: "record_2_failed_001.npz", 4: "record_4_failed_001.npz"}


def write_json(name: str, value: dict) -> None:
    (HERE / name).write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (HERE / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_saved(record: int) -> dict:
    path = FAILED / INPUT_FILES[record]
    with np.load(path, allow_pickle=False) as z:
        data = {k: np.asarray(z[k]) for k in z.files}
    return {"x": data["x"], "y": data["y"], "weights": data["responsibility"], "D": csr_matrix(data["difference"]), "source": str(path), "n": int(data["x"].shape[0])}


def initial_responsibilities(x: np.ndarray, y: np.ndarray, seed: int = SEED) -> np.ndarray:
    """Reproduce the first E-step in fit_lemna before the first weighted solve."""
    rng = np.random.default_rng(seed)
    coef = rng.normal(0.0, 1e-3, (2, x.shape[1]))
    scales = np.full(2, max(float(y.std()), 1e-5))
    weights = np.full(2, 0.5)
    means = x @ coef.T
    standardized = (y[:, None] - means) / scales[None, :]
    logp = (np.log(weights)[None, :] - np.log(scales)[None, :]
            - 0.5 * standardized**2 - 0.5 * np.log(2.0 * np.pi))
    shifted = logp - logp.max(axis=1, keepdims=True)
    resp = np.exp(shifted)
    return resp / resp.sum(axis=1, keepdims=True)


def rebuild_3000(record: int, saved_1000: dict, context: dict) -> dict:
    row = record
    neighborhood_raw = generate_raw_neighborhood(
        context["x_train_raw"][row], context["baseline"], 3000, SEED + record
    )
    x = context["scaler"].transform(neighborhood_raw)
    y = context["model"].predict_proba(x)[:, 1]
    weights = initial_responsibilities(x, y)
    if record == 2:
        raw1000 = generate_raw_neighborhood(
            context["x_train_raw"][row], context["baseline"], 1000, SEED + record
        )
        x1000 = context["scaler"].transform(raw1000)
        y1000 = context["model"].predict_proba(x1000)[:, 1]
        w1000 = initial_responsibilities(x1000, y1000)
        checks = {
            "x_max_abs_diff": float(np.max(np.abs(x1000 - saved_1000["x"]))),
            "y_max_abs_diff": float(np.max(np.abs(y1000 - saved_1000["y"]))),
            "responsibility_max_abs_diff": float(np.max(np.abs(w1000[:, 0] - saved_1000["weights"]))),
        }
        if (checks["x_max_abs_diff"] > 1e-10 or checks["y_max_abs_diff"] > 1e-10
                or checks["responsibility_max_abs_diff"] > 1e-8):
            raise RuntimeError(f"Reconstructed n=1000 Run A inputs do not match saved input: {checks}")
    else:
        checks = {"n1000_input_match": "not independently reconstructed for record 4"}
    return {"x": x, "y": y, "weights": weights[:, 0], "D": saved_1000["D"],
            "source": "reconstructed training-only neighborhood; same record index, seed, baseline, scaler, model, graph",
            "n": 3000, "reconstruction_check": checks}


def weighted_system(data: dict, D: csr_matrix | None = None):
    x = np.asarray(data["x"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    w = np.asarray(data["weights"], dtype=np.float64)
    D = data["D"] if D is None else D
    denom = max(float(w.sum()), 1e-12)
    xw = x * np.sqrt(w[:, None] / denom)
    yw = y * np.sqrt(w / denom)
    A = vstack((csr_matrix(xw), np.sqrt(RHO) * D), format="csr")
    return x, y, w, D, denom, xw, yw, A


def svd_report(data: dict, D: csr_matrix | None = None) -> dict:
    x, y, w, D, denom, xw, yw, A_sparse = weighted_system(data, D)
    A = A_sparse.toarray()
    start = time.perf_counter()
    singular = np.linalg.svd(A, compute_uv=False, full_matrices=False)
    elapsed = time.perf_counter() - start
    sigma_max = float(singular[0])
    sigma_min = float(singular[-1])
    tau = max(A.shape) * np.finfo(np.float64).eps * sigma_max
    thresholds = {"tau": tau, "relative_1e-12": 1e-12 * sigma_max, "relative_1e-10": 1e-10 * sigma_max}
    spectrum = {}
    for name, threshold in thresholds.items():
        retained = singular[singular > threshold]
        spectrum[name] = {
            "threshold": float(threshold), "rank": int(retained.size),
            "effective_condition_retained": float(sigma_max / retained[-1]) if retained.size else None,
        }
    return {"A_shape": list(A.shape), "sigma_max": sigma_max, "sigma_min_returned": sigma_min,
            "thresholds": spectrum, "svd_algorithm": "numpy.linalg.svd(compute_uv=False, full_matrices=False)",
            "full_spectrum_count": int(singular.size), "svd_seconds": elapsed}


def solve_admm(data: dict, max_outer: int, abs_tol: float = ABS_TOL, rel_tol: float = REL_TOL,
               D_override: csr_matrix | None = None) -> dict:
    start_all = time.perf_counter()
    x, y, w, D, denom, xw, yw, A = weighted_system(data, D_override)
    n_features = x.shape[1]
    beta = np.zeros(n_features)
    z = np.zeros(D.shape[0])
    u = np.zeros(D.shape[0])
    previous_z = z.copy()
    gram = None
    factor = None
    if METHOD == "cholesky_jitter":
        gram = xw.T @ xw + RHO * (D.T @ D).toarray()
        factor = cho_factor(gram + 1e-10 * np.eye(n_features), lower=True, check_finite=False)
    residual_history = []
    inner_counts = []
    inner_limit_count = 0
    for it in range(1, max_outer + 1):
        if METHOD == "cholesky_jitter":
            rhs = xw.T @ yw + RHO * np.asarray(D.T @ (z - u)).reshape(-1)
            beta = cho_solve(factor, rhs, check_finite=False)
            inner_counts.append(1)
        else:
            b = np.concatenate((yw, np.sqrt(RHO) * (z - u)))
            if METHOD == "lsmr":
                out = lsmr(A, b, atol=1e-8, btol=1e-8, conlim=1e12, maxiter=INNER_MAXITER, x0=beta)
                beta, stop, count = out[0], int(out[1]), int(out[2])
            else:
                out = lsqr(A, b, atol=1e-8, btol=1e-8, conlim=1e12, iter_lim=INNER_MAXITER, x0=beta)
                beta, stop, count = out[0], int(out[1]), int(out[2])
            inner_counts.append(count)
            inner_limit_count += int(stop == 7)
        db = np.asarray(D @ beta).reshape(-1)
        previous_z = z.copy()
        v = db + u
        abs_v = np.abs(v)
        if float(abs_v.sum()) <= S:
            z = v.copy()
        else:
            ordered = np.sort(abs_v)[::-1]
            csum = np.cumsum(ordered)
            idx = np.arange(1, len(v) + 1)
            valid = ordered - (csum - S) / idx > 0
            last = np.flatnonzero(valid)[-1]
            threshold = (csum[last] - S) / (last + 1)
            z = np.sign(v) * np.maximum(abs_v - threshold, 0.0)
        u += db - z
        primal = float(np.linalg.norm(db - z))
        dual = float(np.linalg.norm(RHO * (D.T @ (z - previous_z))))
        ptol = float(np.sqrt(D.shape[0]) * abs_tol + rel_tol * max(np.linalg.norm(db), np.linalg.norm(z)))
        dtol = float(np.sqrt(n_features) * abs_tol + rel_tol * np.linalg.norm(RHO * (D.T @ u)))
        residual_history.append({"iteration": it, "primal": primal, "dual": dual,
                                 "primal_tolerance": ptol, "dual_tolerance": dtol})
        if primal <= ptol and dual <= dtol:
            break
    db = np.asarray(D @ beta).reshape(-1)
    objective = float(0.5 * np.sum(w * (x @ beta - y) ** 2) / denom)
    return {
        "method": METHOD, "outer_iterations": len(residual_history), "outer_cap": max_outer,
        "converged_residual_certificate": bool(primal <= ptol and dual <= dtol),
        "primal_residual": primal, "dual_residual": dual,
        "primal_tolerance": ptol, "dual_tolerance": dtol,
        "raw_beta_tv": float(np.linalg.norm(db, ord=1)), "excess_tv": float(np.linalg.norm(db, ord=1) - S),
        "tv_z": float(np.linalg.norm(z, ord=1)), "split_gap_l2": float(np.linalg.norm(db - z)),
        "objective": objective, "runtime_seconds": time.perf_counter() - start_all,
        "inner_solver_maxiter": None if METHOD == "cholesky_jitter" else INNER_MAXITER,
        "inner_iteration_limit_stops": None if METHOD == "cholesky_jitter" else inner_limit_count,
        "inner_iterations_min": min(inner_counts) if inner_counts else None,
        "inner_iterations_median": float(np.median(inner_counts)) if inner_counts else None,
        "inner_iterations_max": max(inner_counts) if inner_counts else None,
        "residual_history": residual_history,
    }


def run_methods(data: dict, cap: int, abs_tol: float = ABS_TOL, rel_tol: float = REL_TOL,
                D_override: csr_matrix | None = None) -> list[dict]:
    global METHOD
    rows = []
    for method in METHODS:
        METHOD = method
        rows.append(solve_admm(data, cap, abs_tol, rel_tol, D_override))
    return rows


def setup_training_context() -> dict:
    t0 = time.perf_counter()
    x, y = load_labeled_training(900)
    x_train_raw, _, y_train, _ = train_test_split(x, y, test_size=0.4, stratify=y, random_state=SEED)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    baseline = np.median(x_train_raw, axis=0)
    model = LGBMClassifier(n_estimators=30, num_leaves=31, learning_rate=0.05,
                           objective="binary", random_state=SEED, verbosity=-1)
    model.fit(x_train, y_train)
    return {"x_train_raw": x_train_raw, "baseline": baseline, "scaler": scaler,
            "model": model, "setup_seconds": time.perf_counter() - t0}


def test4(context: dict) -> tuple[dict, list[dict]]:
    started = time.perf_counter()
    records = []
    csv_rows = []
    status = "DONE"
    exact_error = None
    try:
        # Required order: record 2 first. Record 4 is admitted only if record 2 completes under budget.
        for record in (2, 4):
            record_start = time.perf_counter()
            baseline = load_saved(record)
            larger = rebuild_3000(record, baseline, context)
            entries = []
            for nlabel, data in (("n1000", baseline), ("n3000", larger)):
                svd = svd_report(data)
                solved = run_methods(data, 200)
                for sol in solved:
                    entry = {"record_index": record, "subproblem": INPUT_FILES[record], "n": data["n"],
                             "source": data["source"], "rank_condition": svd, **sol}
                    entries.append(entry)
                    csv_rows.append({k: v for k, v in entry.items() if k != "residual_history"})
            elapsed = time.perf_counter() - record_start
            records.append({"record_index": record, "status": "DONE", "elapsed_seconds": elapsed,
                            "reconstruction_check": larger.get("reconstruction_check"), "results": entries})
            if record == 2 and elapsed > BRANCH_BUDGET_SECONDS:
                records.append({"record_index": 4, "status": "NOT RUN",
                                "reason": f"Record 2 took {elapsed:.3f}s, exceeding the {BRANCH_BUDGET_SECONDS:.0f}s per-record admission budget."})
                break
    except Exception as exc:
        status = "BLOCKED" if not records else "PARTIAL"
        exact_error = f"{type(exc).__name__}: {exc}"
        records.append({"record_index": record if 'record' in locals() else None,
                        "status": status, "exact_error": exact_error,
                        "traceback": traceback.format_exc()})
        if 'record' in locals() and record == 2:
            records.append({"record_index": 4, "status": "NOT RUN", "reason": "Record 2 branch did not finish."})
    result = {"test": "TEST 4 — LARGER NEIGHBOURHOOD", "status": status if status != "DONE" else "DONE",
              "configuration": {"rho": RHO, "S": S, "abs_tolerance": ABS_TOL, "relative_tolerance": REL_TOL,
                                "outer_cap": 200, "inner_solver_maxiter": INNER_MAXITER,
                                "changed_only": "neighbourhood sample count n=1000 to n=3000"},
              "record_results": records, "elapsed_seconds": time.perf_counter() - started,
              "branch_budget_seconds": BRANCH_BUDGET_SECONDS, "error": exact_error}
    return result, csv_rows


def test5() -> tuple[dict, list[dict]]:
    started = time.perf_counter()
    records = []
    rows = []
    try:
        edges = ember_fusion_edges()
        touched = np.asarray(sorted({j for e in edges for j in e}), dtype=int)
        if len(touched) != 608:
            raise RuntimeError(f"Expected 608 graph-touched coordinates, found {len(touched)}")
        for record in RECORDS:
            full = load_saved(record)
            reduced_D = full["D"][:, touched]
            scenarios = (("all_2381_coordinates", full, full["D"]),
                         ("graph_touched_608_coordinates", {**full, "x": full["x"][:, touched]}, reduced_D))
            group = []
            for label, data, D in scenarios:
                svd = svd_report(data, D)
                for sol in run_methods(data, 200, D_override=D):
                    entry = {"record_index": record, "subproblem": INPUT_FILES[record], "feature_space": label,
                             "feature_count": data["x"].shape[1], "edge_count": int(D.shape[0]),
                             "changed_feature_space": label != "all_2381_coordinates",
                             "weights_policy": "fixed saved Run A row responsibilities for both feature spaces",
                             "rank_condition": svd, **sol}
                    group.append(entry)
                    rows.append({k: v for k, v in entry.items() if k != "residual_history"})
            records.append({"record_index": record, "status": "DONE", "results": group})
    except Exception as exc:
        records.append({"status": "BLOCKED", "exact_error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()})
    status = "DONE" if len(records) == 2 and all(r.get("status") == "DONE" for r in records) else "BLOCKED"
    return ({"test": "TEST 5 — CONNECTED-FEATURE ABLATION", "status": status,
             "feature_space_note": "608-coordinate ablation is a changed feature space and a different optimisation problem.",
             "configuration": {"rho": RHO, "S": S, "abs_tolerance": ABS_TOL, "relative_tolerance": REL_TOL,
                               "outer_cap": 200, "inner_solver_maxiter": INNER_MAXITER,
                               "full_feature_count": 2381, "graph_touched_feature_count": 608},
             "record_results": records, "elapsed_seconds": time.perf_counter() - started}, rows)


def test6() -> tuple[dict, list[dict]]:
    started = time.perf_counter()
    results = []
    rows = []
    not_run = []
    status = "DONE"
    error = None
    cases = [(200, ABS_TOL, REL_TOL), (1000, ABS_TOL, REL_TOL),
             (2000, ABS_TOL, REL_TOL), (1000, 1e-5, 1e-4)]
    try:
        for record in RECORDS:
            data = load_saved(record)
            for cap, at, rt in cases:
                case_start = time.perf_counter()
                solved = run_methods(data, cap, at, rt)
                for sol in solved:
                    entry = {"record_index": record, "subproblem": INPUT_FILES[record],
                             "outer_cap": cap, "abs_tolerance": at, "relative_tolerance": rt, **sol}
                    results.append(entry)
                    rows.append({k: v for k, v in entry.items() if k != "residual_history"})
                case_elapsed = time.perf_counter() - case_start
                if time.perf_counter() - started > BRANCH_BUDGET_SECONDS:
                    status = "NOT RUN (runtime budget reached)"
                    remaining_records = list(RECORDS[RECORDS.index(record):])
                    remaining_cases = cases[cases.index((cap, at, rt)) + 1:]
                    not_run = [{"record_index": rr, "outer_cap": cc, "abs_tolerance": aa, "relative_tolerance": rr_tol}
                               for rr in remaining_records for cc, aa, rr_tol in remaining_cases]
                    break
            if status != "DONE":
                break
    except Exception as exc:
        status = "BLOCKED"
        error = f"{type(exc).__name__}: {exc}"
        not_run = [{"remaining_cases": "stopped after first branch error"}]
    return ({"test": "TEST 6 — ITERATION / TOLERANCE SENSITIVITY", "status": status,
             "configuration": {"rho": RHO, "S": S, "objective": "0.5 * sum(w*(X*beta-y)^2)/sum(w)",
                               "graph": "unchanged 605-edge EMBER fusion graph", "inner_solver_maxiter": INNER_MAXITER,
                               "inner_limit_stop_code": 7, "branch_budget_seconds": BRANCH_BUDGET_SECONDS,
                               "cases": [{"outer_cap": c, "abs_tolerance": a, "relative_tolerance": r} for c, a, r in cases]},
             "results": results, "not_run": not_run, "error": error,
             "elapsed_seconds": time.perf_counter() - started}, rows)


def main() -> None:
    runtime = {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
               "platform": platform.platform(), "processor": platform.processor(),
               "logical_cpu_count": os.cpu_count(), "training_data_only": True,
               "official_test_records_accessed": False, "malware_executed": False,
               "script": str(Path(__file__).resolve()), "run_a_edited": False}
    ctx_start = time.perf_counter()
    context = setup_training_context()
    runtime["training_model_setup_seconds"] = context["setup_seconds"]
    runtime["total_script_seconds_before_tests"] = time.perf_counter() - ctx_start
    results = {}
    test4_result, test4_rows = test4(context)
    results["test4"] = test4_result
    write_json("test4_larger_neighbourhood.json", test4_result)
    write_csv("test4_larger_neighbourhood.csv", test4_rows)
    test5_result, test5_rows = test5()
    results["test5"] = test5_result
    write_json("test5_ablation.json", test5_result)
    write_csv("test5_ablation.csv", test5_rows)
    test6_result, test6_rows = test6()
    results["test6"] = test6_result
    write_json("test6_sensitivity.json", test6_result)
    write_csv("test6_sensitivity.csv", test6_rows)
    runtime["total_wall_seconds"] = time.perf_counter() - ctx_start
    write_json("run_metadata.json", runtime)
    statuses = "\n".join(f"- {key.upper()}: {value['status']}" for key, value in results.items())
    readme = f"""# Run B — bounded controlled tests

Generated outputs: `{HERE}`

## DONE / NOT RUN / BLOCKED
{statuses}

## Data and controls

- Used only the two saved first-entry weighted subproblems from records 2 and 4. The larger-neighborhood branch started with record 2; record 4 was run only if record 2 completed within the 900-second per-record admission budget.
- Test 4 regenerated neighborhoods through the inspected `generate_raw_neighborhood` implementation, using the same training row, training-median baseline, StandardScaler fitted on the same training split, LGBM model, seed (`SEED + record_index`), mask probability, fusion graph, rho, S and tolerances. Only requested `samples` changed from 1,000 to 3,000. Before accepting n=3,000, the script reconstructed n=1,000 record 2 and compared X, predictions, and initial E-step responsibilities with the saved Run A input.
- Test 5 keeps saved row responsibilities fixed and compares all 2,381 coordinates with the 608 coordinates touched by the 605 fusion edges. The reduced case changes the feature space and therefore is a different optimisation problem.
- Test 6 uses the persisted Run A data, original tolerances at caps 200, 1,000 and 2,000, plus abs=1e-5 / rel=1e-4 at cap 1,000. Inner LSMR/LSQR iteration-limit stops (SciPy stop code 7, max iteration limit) are reported separately from outer ADMM caps and residual certificates. Cholesky's inner solve is direct and has no iterative limit.
- The weighted least-squares system uses normalized weights and A = vstack(X_w, sqrt(rho) D). Full singular spectra are computed with NumPy dense SVD; rank/condition values are reported at tau=max(m,n)*eps*sigma_max, and relative thresholds 1e-12 and 1e-10.
- No raw Run A files were written. No official test records were accessed; no malware was executed; no paper, GitHub repository, or Zenodo record was changed or created.

## Commands and runtime

Command: `experiments\\venv\\Scripts\\python.exe {Path(__file__).name}`

Python: `{sys.version.split()[0]}`; NumPy `{np.__version__}`; SciPy `{scipy.__version__}`. Platform: `{platform.platform()}`; processor: `{platform.processor()}`; logical CPUs: `{os.cpu_count()}`. Runtime details are in `run_metadata.json`; solver settings and exact errors/statuses are in the individual JSON files.

## Artifacts

- `test4_larger_neighbourhood.json/.csv`
- `test5_ablation.json/.csv`
- `test6_sensitivity.json/.csv`
- `run_b.py` (exact script used)
- `run_metadata.json`
"""
    (HERE / "README_RUN_B.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"statuses": {k: v["status"] for k, v in results.items()},
                      "output_folder": str(HERE), "runtime_seconds": runtime["total_wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
