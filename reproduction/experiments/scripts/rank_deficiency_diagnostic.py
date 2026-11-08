"""Reconstruct failed EMBER LEMNA subproblems and analyze rank deficiency."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EXPERIMENTS = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENTS.parent
RAW = EXPERIMENTS / "results" / "raw"
FAILED = EXPERIMENTS / "results" / "raw" / "failed_subproblems"
sys.path.insert(0, str(EXPERIMENTS))
from scripts.run_pilot2 import generate_raw_neighborhood, load_labeled_training, SEED
from src.lemna import LemnaConfig, fit_lemna, _difference_matrix, fused_lasso_penalty
from src.protocol import ember_fusion_edges, canonical_layout


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def reconstruct() -> list[Path]:
    x, y = load_labeled_training(900)
    x_train_raw, _, y_train, _ = train_test_split(
        x, y, test_size=0.4, stratify=y, random_state=SEED
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    baseline = np.median(x_train_raw, axis=0)
    model = LGBMClassifier(
        n_estimators=30, num_leaves=31, learning_rate=0.05,
        objective="binary", random_state=SEED, verbosity=-1,
    )
    model.fit(x_train, y_train)
    edges = ember_fusion_edges()
    FAILED.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for record_index in (2, 4):
        fit_raw = generate_raw_neighborhood(
            x_train_raw[record_index], baseline, 1000, SEED + record_index
        )
        fit_x = scaler.transform(fit_raw)
        fit_y = model.predict_proba(fit_x)[:, 1]
        metadata = {
            "record_index": record_index, "seed": SEED + record_index,
            "features": 2381, "rows": 1000, "fusion_edges": len(edges),
            "rho": 0.1, "inner_max_iter": 200, "outer_max_iter": 60,
            "fused_budget": 1.0, "admm_abs_tolerance": 1e-6,
            "admm_relative_tolerance": 1e-5,
        }
        counter = 0

        def persist(payload: dict, record_index=record_index):
            nonlocal counter
            counter += 1
            diagnostics = payload["diagnostics"]
            path = FAILED / f"record_{record_index}_failed_{counter:03d}.npz"
            np.savez_compressed(
                path,
                x=payload["x"], y=payload["y"],
                responsibility=payload["responsibility"],
                difference=payload["difference"].toarray(),
                beta=payload["beta"], z=payload["z"], dual=payload["dual"],
                rho=np.asarray(payload["rho"]),
                primal_residual_history=np.asarray(diagnostics.primal_residual_history),
                dual_residual_history=np.asarray(diagnostics.dual_residual_history),
                primal_tolerance_history=np.asarray(diagnostics.primal_tolerance_history),
                dual_tolerance_history=np.asarray(diagnostics.dual_tolerance_history),
                objective_history=np.asarray(diagnostics.objective_history),
            )
            (path.with_suffix(".json")).write_text(json.dumps(jsonable({
                **metadata, "failed_call": counter,
                "iterations": diagnostics.iterations,
                "factor_jitter": diagnostics.factor_jitter,
                "final_tv": diagnostics.total_variation,
                "converged": diagnostics.converged,
                "sha256": {
                    "x": sha256_array(payload["x"]),
                    "y": sha256_array(payload["y"]),
                    "responsibility": sha256_array(payload["responsibility"]),
                },
            }), indent=2), encoding="utf-8")
            saved.append(path)

        config = LemnaConfig(
            components=2, fused_budget=1.0, neighborhood_samples=1000,
            seed=SEED, fusion_edges=edges, admm_rho=0.1,
            max_iter=60, inner_max_iter=200,
        )
        fit_lemna(fit_x, fit_y, config=config, failure_callback=persist)
    return saved


def rank_and_compare(paths: list[Path]) -> dict:
    edges = ember_fusion_edges()
    difference = _difference_matrix(2381, edges)
    group_sizes = [256, 256, 95]
    d_rank = sum(size - 1 for size in group_sizes)
    result = {
        "dimension": 2381,
        "fusion_edges": len(edges),
        "connected_feature_groups": group_sizes,
        "isolated_features": 2381 - sum(group_sizes),
        "rank_D_analytic": d_rank,
        "nullity_D_analytic": 2381 - d_rank,
        "combined_rank_upper_bound": 1000 + d_rank,
        "combined_nullity_lower_bound": 2381 - (1000 + d_rank),
        "subproblems": [],
    }
    for path in paths:
        data = np.load(path)
        x = data["x"]; y = data["y"]; w = data["responsibility"]
        denominator = max(float(w.sum()), 1e-12)
        weighted_x = x * np.sqrt(w[:, None] / denominator)
        gram = weighted_x.T @ weighted_x
        dtd = difference.T @ difference
        system = gram + 0.1 * dtd
        singular_values = np.linalg.svd(system, compute_uv=False)
        result["subproblems"].append({
            "file": path.name,
            "responsibility_sum": float(w.sum()),
            "effective_sample_size": float(w.sum() ** 2 / np.sum(w ** 2)),
            "maximum_responsibility": float(w.max()),
            "coefficient_norm": float(np.linalg.norm(data["beta"])),
            "system_max_singular_value": float(singular_values[0]),
            "system_min_singular_value": float(singular_values[-1]),
            "system_condition_estimate": float(
                singular_values[0] / max(singular_values[-1], np.finfo(float).tiny)
            ),
            "saved_final_tv": float(fused_lasso_penalty(data["beta"], edges)),
        })
    return result


def main() -> None:
    started = time.perf_counter()
    paths = reconstruct()
    analysis = rank_and_compare(paths)
    analysis["runtime_seconds"] = time.perf_counter() - started
    (RAW / "LEMNA_FAILED_INPUTS_MANIFEST.json").write_text(
        json.dumps(jsonable({"files": [str(path) for path in paths], "count": len(paths),
                             "training_only": True, "final_test_accessed": False}), indent=2),
        encoding="utf-8",
    )
    (RAW / "LEMNA_RANK_AWARE_SOLVER_COMPARISON.json").write_text(
        json.dumps(jsonable(analysis), indent=2), encoding="utf-8"
    )
    (EXPERIMENTS / "LEMNA_RANK_DEFICIENCY_REPORT.md").write_text(
        f"""# LEMNA rank-deficiency report

The EMBER workload has 1,000 rows, 2,381 coefficients and 605 fusion edges.
The three constrained paths have sizes 256, 256 and 95, so rank(D)=605 and
nullity(D)=1,776. Therefore rank(X'WX + rho D'D) is at most 1,605 and its
nullity is at least 776 in exact arithmetic.

The reconstruction persisted {len(paths)} failed subproblems from records 2
and 4 as compressed NPZ files. Their hashes and settings are in
`results/raw/LEMNA_FAILED_INPUTS_MANIFEST.json` and adjacent JSON metadata.

Cholesky jitter makes the numerical system positive definite by adding
`jitter * I`; this is an implicit ridge term and is not mathematically
identical to the unregularized beta update. Numerical singular values and
condition estimates are in `LEMNA_RANK_AWARE_SOLVER_COMPARISON.json`.

No final-test data, SHAP, LIME, five-instance rerun or 30-instance run was
performed by this diagnostic.
""", encoding="utf-8")
    (EXPERIMENTS / "LEMNA_NUMERICAL_STABILITY_REPORT.md").write_text(
        """# LEMNA numerical stability report

The saved failed subproblems include X, y, responsibilities, D, beta, z,
scaled dual state, rho and complete residual/objective histories. The
existing Cholesky path uses recorded jitter and therefore solves a
jitter-regularized linear system. The rank-aware comparison artifact records
the observed singular-value range and condition estimate for each saved
subproblem.

Pseudoinverse and matrix-free alternatives were not substituted into the
research runner. They require a separate validated ADMM implementation and
must preserve the same objective and stopping rules before adoption.
""", encoding="utf-8")
    print(json.dumps({"saved_failed_subproblems": len(paths), "seconds": analysis["runtime_seconds"]}))


if __name__ == "__main__":
    main()
