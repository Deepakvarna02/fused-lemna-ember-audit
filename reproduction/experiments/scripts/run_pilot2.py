"""Training-only scaled-model and LEMNA technical pilot.

This script never opens test_features.jsonl. It records errors rather than
turning incomplete measurements into successful results.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import shap
from ember.features import PEFeatureExtractor
from lime.lime_tabular import LimeTabularExplainer
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Make the experiments package importable when this file is executed directly.
EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from src.lemna import (
    LemnaConfig,
    _weighted_penalized_regression,
    explain_black_box,
    fit_lemna,
    fused_lasso_penalty,
    generate_mask_neighborhood,
)
from src.protocol import ember_fusion_edges


ROOT = EXPERIMENTS.parent
DEFAULT_OUTPUT = ROOT / "experiments" / "results" / "raw" / "pilot2_results.json"
SEED = 20261106
FEATURES = 2381


def load_labeled_training(limit_per_class: int) -> tuple[np.ndarray, np.ndarray]:
    extractor = PEFeatureExtractor(feature_version=2, print_feature_warning=False)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    counts = {0: 0, 1: 0}
    for path in sorted(ROOT.glob("train_features_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                label = record["label"]
                if label not in counts or counts[label] >= limit_per_class:
                    continue
                rows.append(extractor.process_raw_features(record))
                labels.append(label)
                counts[label] += 1
                if all(value == limit_per_class for value in counts.values()):
                    x = np.asarray(rows, dtype=np.float64)
                    y = np.asarray(labels, dtype=np.int8)
                    if x.shape != (2 * limit_per_class, FEATURES):
                        raise RuntimeError(f"Unexpected training shape: {x.shape}")
                    return x, y
    raise RuntimeError(f"Insufficient labeled training data: {counts}")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def generate_raw_neighborhood(
    instance: np.ndarray,
    baseline: np.ndarray,
    samples: int,
    seed: int,
    mask_probability: float = 0.5,
) -> np.ndarray:
    """Nullify raw-space features to a training-only baseline.

    Zero is not assumed to be a valid absence value. The baseline is the
    coordinate-wise training median; this remains an off-manifold feature-space
    approximation and is recorded as such in the pilot output.
    """
    if samples < 2 or not 0.0 <= mask_probability <= 1.0:
        raise ValueError("Invalid raw neighborhood settings")
    instance = np.asarray(instance, dtype=np.float64).reshape(-1)
    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    if instance.shape != baseline.shape:
        raise ValueError("Instance and baseline dimensions do not match")
    rng = np.random.default_rng(seed)
    keep = rng.random((samples - 1, instance.size)) >= mask_probability
    neighborhood = np.empty((samples, instance.size), dtype=np.float64)
    neighborhood[0] = instance
    neighborhood[1:] = np.where(keep, instance[None, :], baseline[None, :])
    return neighborhood


def model_metrics(model, x_hold, y_hold, fit_seconds: float) -> dict:
    probabilities = np.asarray(model.predict_proba(x_hold))[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int8)
    return {
        "fit_seconds": fit_seconds,
        "accuracy": float(accuracy_score(y_hold, predictions)),
        "roc_auc_from_probabilities": float(roc_auc_score(y_hold, probabilities)),
    }


def lime_vector(explanation, dimension: int) -> tuple[np.ndarray, float]:
    vector = np.zeros(dimension, dtype=np.float64)
    label = 1 if 1 in explanation.local_exp else next(iter(explanation.local_exp))
    for index, coefficient in explanation.local_exp[label]:
        vector[int(index)] = coefficient
    intercept = float(explanation.intercept[label])
    return vector, intercept


def explanation_record(
    model, x_train_scaled, instance_raw, baseline_raw, index, config, scaler, lime_samples
):
    fit_raw = generate_raw_neighborhood(
        instance_raw, baseline_raw, config.neighborhood_samples, config.seed + index
    )
    heldout_raw = generate_raw_neighborhood(
        instance_raw, baseline_raw, config.neighborhood_samples, config.seed + 10000 + index
    )
    neighborhood = scaler.transform(fit_raw)
    heldout = scaler.transform(heldout_raw)
    neighborhood_outputs = model.predict_proba(neighborhood)[:, 1]
    heldout_outputs = model.predict_proba(heldout)[:, 1]
    instance = scaler.transform(instance_raw.reshape(1, -1))[0]
    target_output = float(model.predict_proba(instance.reshape(1, -1))[0, 1])

    started = time.perf_counter()
    lemna = fit_lemna(
        neighborhood,
        neighborhood_outputs,
        config=config,
        target=instance,
        target_output=target_output,
    )
    lemna_seconds = time.perf_counter() - started

    lime_explainer = LimeTabularExplainer(
        x_train_scaled,
        mode="classification",
        class_names=["benign", "malicious"],
        random_state=config.seed + index,
        discretize_continuous=False,
        feature_selection="none",
    )
    started = time.perf_counter()
    lime = lime_explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=FEATURES,
        num_samples=lime_samples,
        labels=(1,),
    )
    lime_seconds = time.perf_counter() - started
    lime_attribution, lime_intercept = lime_vector(lime, FEATURES)

    lemna_prediction = heldout @ lemna.attribution
    lime_prediction = lime_intercept + heldout @ lime_attribution
    shap_started = time.perf_counter()
    shap_values = shap.TreeExplainer(model)(
        instance.reshape(1, -1)
    ).values
    shap_seconds = time.perf_counter() - shap_started
    if shap_values.ndim == 3:
        shap_attribution = np.asarray(shap_values[0, :, -1], dtype=np.float64)
    else:
        shap_attribution = np.asarray(shap_values[0], dtype=np.float64)
    all_admm_converged = all(
        item.converged for item in lemna.optimizer_diagnostics
    )
    top10 = np.argsort(-np.abs(lemna.attribution), kind="stable")[:10]
    return {
        "record_index": index,
        "successful": bool(lemna.converged and all_admm_converged),
        "shap": {
            "runtime_seconds": shap_seconds,
            "attribution": shap_attribution.tolist(),
            "top5_feature_indices": np.argsort(-np.abs(shap_attribution))[:5].tolist(),
            "top10_feature_indices": np.argsort(-np.abs(shap_attribution))[:10].tolist(),
        },
        "lemna": {
            "runtime_seconds": lemna_seconds,
            "local_fidelity_rmse": float(
                mean_squared_error(heldout_outputs, lemna_prediction) ** 0.5
            ),
            "converged": bool(lemna.converged),
            "iterations": int(lemna.iterations),
            "selected_component": int(lemna.selected_component),
            "top10_feature_indices": top10.tolist(),
            "coefficients": lemna.attribution.tolist(),
            "optimizer_converged": [
                bool(item.converged) for item in lemna.optimizer_diagnostics
            ],
            "optimizer_diagnostics": [
                {
                    "iterations": item.iterations,
                    "converged": item.converged,
                    "primal_residual": item.primal_residual,
                    "dual_residual": item.dual_residual,
                    "primal_tolerance": item.primal_tolerance,
                    "dual_tolerance": item.dual_tolerance,
                    "objective_history": list(item.objective_history),
                    "primal_residual_history": list(item.primal_residual_history),
                    "dual_residual_history": list(item.dual_residual_history),
                    "rho_history": list(item.rho_history),
                    "final_tv": item.total_variation,
                    "termination_status": item.termination_status,
                }
                for item in lemna.optimizer_diagnostics
            ],
        },
        "lime": {
            "runtime_seconds": lime_seconds,
            "local_fidelity_rmse": float(
                mean_squared_error(heldout_outputs, lime_prediction) ** 0.5
            ),
            "top10_feature_indices": np.argsort(-np.abs(lime_attribution))[:10].tolist(),
            "attribution": lime_attribution.tolist(),
        },
        "shared_evaluation_neighborhood": {
            "fit_rows": int(neighborhood.shape[0]),
            "heldout_rows": int(heldout.shape[0]),
            "feature_dimension": int(neighborhood.shape[1]),
            "fit_seed": config.seed + index,
            "heldout_seed": config.seed + 10000 + index,
            "raw_space_baseline": "training_coordinate_median",
            "heldout_model_predictions": heldout_outputs.tolist(),
            "lemna_predictions": lemna_prediction.tolist(),
            "lime_predictions": lime_prediction.tolist(),
            "lemna_residuals": (heldout_outputs - lemna_prediction).tolist(),
            "lime_residuals": (heldout_outputs - lime_prediction).tolist(),
        },
    }


def repeatability(model, x_train_scaled, instance_raw, baseline_raw, scaler, config) -> dict:
    fit_raw = generate_raw_neighborhood(
        instance_raw, baseline_raw, config.neighborhood_samples, config.seed
    )
    neighborhood = scaler.transform(fit_raw)
    outputs = model.predict_proba(neighborhood)[:, 1]
    instance = scaler.transform(instance_raw.reshape(1, -1))[0]
    same_left = fit_lemna(neighborhood, outputs, config, instance)
    same_right = fit_lemna(neighborhood, outputs, config, instance)
    different = fit_lemna(
        neighborhood,
        outputs,
        LemnaConfig(**{**config.__dict__, "seed": config.seed + 1}),
        scaler.transform(instance_raw.reshape(1, -1))[0],
    )
    left = set(np.argsort(-np.abs(same_left.attribution))[:10])
    right = set(np.argsort(-np.abs(same_right.attribution))[:10])
    changed = set(np.argsort(-np.abs(different.attribution))[:10])
    return {
        "same_seed_top10_overlap": len(left & right) / 10.0,
        "different_seed_top10_overlap": len(left & changed) / 10.0,
        "same_seed_coefficients_equal": bool(
            np.array_equal(same_left.attribution, same_right.attribution)
        ),
    }


def admm_stress(config, edges) -> dict:
    rng = np.random.default_rng(SEED + 99)
    del rng
    # Identity design plus alternating targets prevents the underdetermined
    # random-design case from making the TV constraint inactive.
    x = np.eye(FEATURES, dtype=np.float64)
    y = np.where(np.arange(FEATURES) % 2 == 0, 1.0, -1.0)
    weights = np.ones(FEATURES)
    tracemalloc.start()
    started = time.perf_counter()
    beta, objective, diagnostics = _weighted_penalized_regression(
        x, y, weights, 1e-3, config.inner_max_iter, edges, "admm",
        config.admm_rho, config.admm_abs_tolerance, config.admm_relative_tolerance,
    )
    runtime = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "features": FEATURES,
        "samples": FEATURES,
        "budget": 1e-3,
        "runtime_seconds": runtime,
        "peak_python_traced_bytes": peak,
        "objective": objective,
        "iterations": diagnostics.iterations,
        "converged": diagnostics.converged,
        "final_tv": fused_lasso_penalty(beta, edges),
        "coefficient_finite": bool(np.isfinite(beta).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-class", type=int, default=900)
    parser.add_argument("--explanation-records", type=int, default=20)
    parser.add_argument("--lime-samples", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-stress", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--admm-rho", type=float, default=0.1)
    parser.add_argument("--inner-max-iter", type=int, default=200)
    parser.add_argument("--adaptive-rho", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    x, y = load_labeled_training(args.limit_per_class)
    x_train_raw, x_hold_raw, y_train, y_hold = train_test_split(
        x, y, test_size=0.4, stratify=y, random_state=SEED
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_raw)
    x_hold_scaled = scaler.transform(x_hold_raw)
    baseline_raw = np.median(x_train_raw, axis=0)
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=50, max_features="sqrt", random_state=SEED, n_jobs=-1
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=30, num_leaves=31, learning_rate=0.05,
            objective="binary", random_state=SEED, verbosity=-1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=20, early_stopping=True,
            random_state=SEED, batch_size=128,
        ),
    }
    results = {
        "pilot": True,
        "training_only": True,
        "final_test_accessed": False,
        "settings": {
            "seed": SEED,
            "records_per_class": args.limit_per_class,
            "lemna": {"K": 2, "S": 1.0, "N": 1000, "mask_probability": 0.5},
            "lime_samples": args.lime_samples,
            "scaling": "StandardScaler fit on x_train only",
            "perturbation": "raw-space coordinate median nullification; scaler applied afterward",
            "fidelity": "separate held-out raw-space neighborhoods",
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "models": {},
    }
    fitted = {}
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(x_train_scaled, y_train)
        fitted[name] = model
        results["models"][name] = model_metrics(
            model, x_hold_scaled, y_hold, time.perf_counter() - t0
        )

    tree = fitted["lightgbm"]
    instance = x_train_scaled[0]
    t0 = time.perf_counter()
    shap_values = shap.TreeExplainer(tree)(instance.reshape(1, -1)).values
    shap_seconds = time.perf_counter() - t0
    if shap_values.ndim == 3:
        shap_shape = list(shap_values.shape)
        shap_top10 = np.argsort(-np.abs(shap_values[0, :, -1]))[:10].tolist()
    else:
        shap_shape = list(shap_values.shape)
        shap_top10 = np.argsort(-np.abs(shap_values[0]))[:10].tolist()
    config = LemnaConfig(
        components=2, fused_budget=1.0, neighborhood_samples=1000,
        seed=SEED, fusion_edges=ember_fusion_edges(), admm_rho=args.admm_rho,
        max_iter=60, inner_max_iter=args.inner_max_iter,
        admm_adaptive_rho=args.adaptive_rho,
    )
    results["single_explanation"] = {
        "lemna_and_lime": explanation_record(
            tree, x_train_scaled, x_train_raw[0], baseline_raw, 0,
            config, scaler, args.lime_samples
        ),
    }
    records = []
    for index in range(
        args.start_index,
        min(args.start_index + args.explanation_records, len(x_train_raw)),
    ):
        try:
            records.append(
                explanation_record(
                    tree, x_train_scaled, x_train_raw[index], baseline_raw,
                    index, config, scaler, args.lime_samples,
                )
            )
        except Exception as error:
            records.append({"record_index": index, "error": repr(error)})
    results["lemna_20_training_records"] = records
    results["repeatability"] = repeatability(
        tree, x_train_scaled, x_train_raw[0], baseline_raw, scaler, config
    )
    if args.run_stress:
        try:
            results["admm_stress"] = admm_stress(config, ember_fusion_edges())
        except Exception as error:
            results["admm_stress"] = {
                "status": "failed",
                "error": repr(error),
                "features": FEATURES,
                "budget": 1e-3,
            }
    results["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output.with_name(args.output.stem + "_EXPLANATIONS.npz"),
        records=np.asarray(_jsonable(records), dtype=object),
    )
    csv_path = args.output.with_name(args.output.stem + "_PER_INSTANCE.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "record_index", "successful", "lemna_converged",
                "lemna_iterations", "lemna_rmse", "lime_rmse",
                "shap_runtime_seconds", "lemna_runtime_seconds",
                "lime_runtime_seconds", "error",
            ],
        )
        writer.writeheader()
        for item in records:
            if "error" in item:
                writer.writerow({"record_index": item.get("record_index"), "error": item["error"]})
                continue
            writer.writerow({
                "record_index": item["record_index"],
                "successful": item["successful"],
                "lemna_converged": item["lemna"]["converged"],
                "lemna_iterations": item["lemna"]["iterations"],
                "lemna_rmse": item["lemna"]["local_fidelity_rmse"],
                "lime_rmse": item["lime"]["local_fidelity_rmse"],
                "shap_runtime_seconds": item["shap"]["runtime_seconds"],
                "lemna_runtime_seconds": item["lemna"]["runtime_seconds"],
                "lime_runtime_seconds": item["lime"]["runtime_seconds"],
                "error": "",
            })
    print(json.dumps({
        "output": str(args.output),
        "elapsed_seconds": results["elapsed_seconds"],
        "final_test_accessed": False,
        "records_completed": sum("error" not in item for item in records),
    }, indent=2))


if __name__ == "__main__":
    main()
