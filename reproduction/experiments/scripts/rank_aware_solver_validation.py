"""Validate rectangular rank-aware beta updates on persisted EMBER failures."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import lsqr, lsmr

EXPERIMENTS = Path(__file__).resolve().parents[1]
RAW = EXPERIMENTS / "results" / "raw"
FAILED = RAW / "failed_subproblems"
sys.path.insert(0, str(EXPERIMENTS))
from src.protocol import ember_fusion_edges

RHO = 0.1
BUDGET = 1.0
ABS_TOL = 1e-6
REL_TOL = 1e-5
MAX_ADMM_ITER = 200
LS_MAXITER = 300


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_l1(value: np.ndarray, radius: float) -> np.ndarray:
    if np.sum(np.abs(value)) <= radius:
        return value.copy()
    absolute = np.abs(value)
    ordered = np.sort(absolute)[::-1]
    cumulative = np.cumsum(ordered)
    indices = np.arange(1, value.size + 1)
    valid = ordered - (cumulative - radius) / indices > 0
    last = np.flatnonzero(valid)[-1]
    threshold = (cumulative[last] - radius) / (last + 1)
    return np.sign(value) * np.maximum(absolute - threshold, 0.0)


def graph_report(edges: tuple[tuple[int, int], ...], dimension: int) -> dict:
    parent = np.arange(dimension)
    degree = np.zeros(dimension, dtype=int)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    duplicate_count = 0
    self_loops = []
    seen: set[tuple[int, int]] = set()
    for left, right in edges:
        if left == right:
            self_loops.append((left, right))
        key = (min(left, right), max(left, right))
        if key in seen:
            duplicate_count += 1
        seen.add(key)
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[left_root] = right_root
        degree[left] += 1
        degree[right] += 1
    groups: dict[int, list[int]] = {}
    for index in range(dimension):
        groups.setdefault(find(index), []).append(index)
    nontrivial = sorted((len(values) for values in groups.values() if len(values) > 1), reverse=True)
    isolated = sum(len(values) == 1 for values in groups.values())
    return {
        "edge_rows": len(edges),
        "unique_undirected_edges": len(seen),
        "duplicate_edges": duplicate_count,
        "self_loops": self_loops,
        "connected_component_sizes": nontrivial,
        "isolated_features": isolated,
        "graph_rank": dimension - len(groups),
        "nullity": len(groups),
        "degree_nonzero_features": int(np.count_nonzero(degree)),
    }


def diagnostics(beta, z, previous_z, dual, difference, x, y, weights, rho):
    primal_vector = difference @ beta - z
    dual_vector = rho * difference.T @ (z - previous_z)
    primal = float(np.linalg.norm(primal_vector))
    dual_norm = float(np.linalg.norm(dual_vector))
    primal_tol = float(
        np.sqrt(difference.shape[0]) * ABS_TOL
        + REL_TOL * max(np.linalg.norm(difference @ beta), np.linalg.norm(z))
    )
    dual_tol = float(
        np.sqrt(difference.shape[1]) * ABS_TOL
        + REL_TOL * np.linalg.norm(rho * difference.T @ dual)
    )
    weighted_loss = float(0.5 * np.sum(weights * (x @ beta - y) ** 2) / max(weights.sum(), 1e-12))
    tv = float(np.sum(np.abs(difference @ beta)))
    return {
        "primal_residual": primal,
        "dual_residual": dual_norm,
        "primal_tolerance": primal_tol,
        "dual_tolerance": dual_tol,
        "weighted_regression_objective": weighted_loss,
        "tv": tv,
        "prediction_norm": float(np.linalg.norm(x @ beta)),
    }


def beta_update(method, weighted_x, weighted_y, difference, z, dual, rho, x0, cholesky_factor=None, gram=None):
    a = vstack((csr_matrix(weighted_x), np.sqrt(rho) * difference), format="csr")
    b = np.concatenate((weighted_y, np.sqrt(rho) * (z - dual)))
    start = time.perf_counter()
    if method == "lsmr":
        result = lsmr(a, b, atol=1e-8, btol=1e-8, conlim=1e12, maxiter=LS_MAXITER, x0=x0)
        beta, istop, itn, normr, normar, anorm, acond, normx = result
        status = {"istop": int(istop), "iterations": int(itn), "residual_norm": float(normr),
                  "normal_residual_norm": float(normar), "estimated_condition": float(acond),
                  "coefficient_norm": float(normx)}
    elif method == "lsqr":
        result = lsqr(a, b, atol=1e-8, btol=1e-8, conlim=1e12, iter_lim=LS_MAXITER, x0=x0)
        beta, istop, itn, normr, _, anorm, acond, _, normx, _ = result
        status = {"istop": int(istop), "iterations": int(itn), "residual_norm": float(normr),
                  "normal_residual_norm": float(result[7]), "estimated_condition": float(acond),
                  "coefficient_norm": float(normx)}
    else:
        if gram is None:
            gram = weighted_x.T @ weighted_x + rho * (difference.T @ difference).toarray()
        rhs = weighted_x.T @ weighted_y + rho * difference.T @ (z - dual)
        if cholesky_factor is None:
            cholesky_factor = cho_factor(
                gram + 1e-10 * np.eye(gram.shape[0]), lower=True, check_finite=False
            )
        beta = cho_solve(cholesky_factor, rhs, check_finite=False)
        status = {"istop": 1, "iterations": 1, "residual_norm": float(np.linalg.norm(a @ beta - b)),
                  "normal_residual_norm": float(np.linalg.norm(gram @ beta - rhs)),
                  "estimated_condition": None, "coefficient_norm": float(np.linalg.norm(beta)),
                  "jitter": 1e-10}
    status["runtime_seconds"] = time.perf_counter() - start
    status["solver_converged"] = bool(status["istop"] in {1, 2})
    return np.asarray(beta), status


def run_admm(method, data, warm_start=False):
    x = np.asarray(data["x"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    weights = np.asarray(data["responsibility"], dtype=np.float64)
    difference = csr_matrix(np.asarray(data["difference"], dtype=np.float64))
    denominator = max(float(weights.sum()), 1e-12)
    weighted_x = x * np.sqrt(weights[:, None] / denominator)
    weighted_y = y * np.sqrt(weights / denominator)
    beta = np.asarray(data["beta"], dtype=np.float64).copy() if warm_start else np.zeros(x.shape[1])
    z = project_l1(np.asarray(difference @ beta).reshape(-1), BUDGET)
    dual = np.zeros(difference.shape[0])
    cholesky_factor = None
    gram = None
    if method == "cholesky_jitter":
        gram = weighted_x.T @ weighted_x + RHO * (difference.T @ difference).toarray()
        cholesky_factor = cho_factor(
            gram + 1e-10 * np.eye(gram.shape[0]), lower=True, check_finite=False
        )
    previous_z = z.copy()
    history = []
    start = time.perf_counter()
    beta_statuses = []
    for iteration in range(1, MAX_ADMM_ITER + 1):
        beta, beta_status = beta_update(
            method, weighted_x, weighted_y, difference, z, dual, RHO, beta,
            cholesky_factor, gram
        )
        beta_statuses.append(beta_status)
        previous_z = z.copy()
        z = project_l1(np.asarray(difference @ beta).reshape(-1) + dual, BUDGET)
        dual += np.asarray(difference @ beta).reshape(-1) - z
        current = diagnostics(beta, z, previous_z, dual, difference, x, y, weights, RHO)
        current["iteration"] = iteration
        history.append(current)
        if current["primal_residual"] <= current["primal_tolerance"] and current["dual_residual"] <= current["dual_tolerance"]:
            break
    final = history[-1]
    return {
        "method": method,
        "warm_start": warm_start,
        "iterations": len(history),
        "converged": bool(final["primal_residual"] <= final["primal_tolerance"]
                          and final["dual_residual"] <= final["dual_tolerance"]
                          and all(item["solver_converged"] for item in beta_statuses)),
        "beta_step_converged_all": bool(all(item["solver_converged"] for item in beta_statuses)),
        "final": final,
        "beta_step": beta_statuses[-1],
        "prediction_difference_from_saved_beta": float(
            np.linalg.norm(x @ beta - x @ np.asarray(data["beta"], dtype=np.float64))
        ),
        "coefficient_norm": float(np.linalg.norm(beta)),
        "runtime_seconds": float(time.perf_counter() - start),
        "raw_tv": final["tv"],
        "returned_tv": final["tv"],
        "history": history,
    }


def main() -> None:
    manifest_path = RAW / "LEMNA_FAILED_INPUTS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [Path(item) for item in manifest["files"]]
    if len(paths) != 16:
        raise RuntimeError(f"Expected 16 persisted inputs, found {len(paths)}")
    edges = ember_fusion_edges()
    graph = graph_report(edges, 2381)
    numerical_ranks = []
    records = []
    for path in paths:
        file_hash = sha256_file(path)
        data = np.load(path)
        required = {"x", "y", "responsibility", "difference", "beta", "z", "dual", "rho"}
        missing = sorted(required - set(data.files))
        if missing:
            raise RuntimeError(f"{path.name} missing {missing}")
        arrays = {name: np.asarray(data[name]) for name in required}
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise RuntimeError(f"{path.name} contains non-finite persisted values")
        if arrays["x"].shape != (1000, 2381) or arrays["difference"].shape != (605, 2381):
            raise RuntimeError(f"Unexpected shapes in {path.name}")
        npz_edges_rank = int(np.linalg.matrix_rank(arrays["difference"], tol=1e-10))
        numerical_ranks.append(npz_edges_rank)
        method_results = {}
        for method in ("cholesky_jitter", "lsmr", "lsqr"):
            method_results[method] = run_admm(method, arrays, warm_start=False)
        records.append({
            "file": path.name,
            "sha256_file": file_hash,
            "record_index": int(json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["record_index"]),
            "difference_numerical_rank": npz_edges_rank,
            "methods": method_results,
        })
    output = {
        "training_only": True,
        "final_test_accessed": False,
        "configuration": {"rho": RHO, "fused_budget": BUDGET, "abs_tolerance": ABS_TOL,
                          "relative_tolerance": REL_TOL, "max_admm_iter": MAX_ADMM_ITER,
                          "linear_solver_maxiter": LS_MAXITER},
        "fusion_graph": graph,
        "saved_difference_numerical_ranks": sorted(set(numerical_ranks)),
        "verified_input_count": len(records),
        "subproblems": records,
    }
    (RAW / "LEMNA_RANK_AWARE_SOLVER_VALIDATION.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    (EXPERIMENTS / "LEMNA_RANK_AWARE_SOLVER_VALIDATION.md").write_text(
        "# Rank-aware LEMNA solver validation\n\n"
        f"Verified {len(records)} exact persisted EMBER failed subproblems. "
        f"The configured graph has {graph['edge_rows']} edge rows, "
        f"{graph['graph_rank']} graph rank, and {graph['nullity']} nullity. "
        "Results for jittered Cholesky, LSMR, and LSQR are in "
        "`results/raw/LEMNA_RANK_AWARE_SOLVER_VALIDATION.json`.\n\n"
        "The rectangular methods solve the augmented least-squares system "
        "without explicitly forming X'WX + rho D'D and without diagonal jitter. "
        "ADMM convergence is reported only when both residual certificates and "
        "all linear-solver status checks pass. Raw TV is reported separately; "
        "no infeasible beta is relabeled feasible by post-hoc correction.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
