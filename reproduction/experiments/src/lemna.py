"""A documented Python 3 implementation of LEMNA's local mixture regression.

The implementation follows Guo et al., CCS 2018: a finite mixture of local
linear regressions is fitted to black-box outputs, with a fused-lasso penalty
on each component's ordered coefficient vector. It is intentionally exposed
as a separate implementation rather than being aliased to SHAP or LIME.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
from scipy.special import logsumexp
from scipy.sparse import csr_matrix, lil_matrix


Array = np.ndarray


@dataclass(frozen=True)
class LemnaConfig:
    components: int = 2
    fused_budget: float = 1.0
    max_iter: int = 30
    inner_max_iter: int = 80
    tolerance: float = 1e-5
    seed: int = 20261112
    min_scale: float = 1e-5
    fusion_edges: tuple[tuple[int, int], ...] | None = None
    neighborhood_samples: int = 1000
    mask_probability: float = 0.5
    optimizer: str = "admm"
    admm_rho: float = 1.0
    admm_abs_tolerance: float = 1e-6
    admm_relative_tolerance: float = 1e-5
    admm_adaptive_rho: bool = False
    admm_rho_mu: float = 10.0
    admm_rho_tau: float = 2.0
    admm_factor_jitter: float | None = None


@dataclass
class LemnaResult:
    coefficients: Array
    intercepts: Array
    scales: Array
    weights: Array
    responsibilities: Array
    log_likelihood: float
    objective: float  # Negative finite-mixture log likelihood.
    attribution: Array
    selected_component: int
    iterations: int
    converged: bool
    optimizer_diagnostics: tuple["OptimizerDiagnostics", ...] = ()
    em_trace: tuple["EMIterationDiagnostics", ...] = ()


@dataclass(frozen=True)
class EMIterationDiagnostics:
    iteration: int
    log_likelihood: float
    absolute_likelihood_change: float
    relative_likelihood_change: float
    weights: tuple[float, ...]
    scales: tuple[float, ...]
    coefficient_changes: tuple[float, ...]
    inner_converged: tuple[bool, ...]
    inner_iterations: tuple[int, ...]
    inner_primal_residuals: tuple[float, ...]
    inner_dual_residuals: tuple[float, ...]
    inner_primal_tolerances: tuple[float, ...]
    inner_dual_tolerances: tuple[float, ...]
    inner_factor_jitters: tuple[float, ...]
    inner_correction_applied: tuple[bool, ...]
    inner_total_variations: tuple[float, ...]


@dataclass(frozen=True)
class OptimizerDiagnostics:
    optimizer: str
    iterations: int
    converged: bool
    primal_residual: float
    dual_residual: float
    primal_tolerance: float
    dual_tolerance: float
    objective_history: tuple[float, ...]
    correction_applied: bool
    total_variation: float
    termination_status: str = "unknown"
    factor_jitter: float = 0.0
    primal_residual_history: tuple[float, ...] = ()
    dual_residual_history: tuple[float, ...] = ()
    primal_tolerance_history: tuple[float, ...] = ()
    dual_tolerance_history: tuple[float, ...] = ()
    rho_history: tuple[float, ...] = ()


def fused_lasso_penalty(beta: Array, edges: tuple[tuple[int, int], ...] | None = None) -> float:
    """Return the component-wise fused-lasso total variation."""
    beta = np.asarray(beta, dtype=np.float64)
    if edges is None:
        return float(fsum(np.abs(np.diff(beta)).tolist()))
    return float(fsum(abs(beta[right] - beta[left]) for left, right in edges))


def _difference_matrix(
    n_features: int, edges: tuple[tuple[int, int], ...]
) -> csr_matrix:
    matrix = lil_matrix((len(edges), n_features), dtype=np.float64)
    for row, (left, right) in enumerate(edges):
        matrix[row, left] = -1.0
        matrix[row, right] = 1.0
    return matrix.tocsr()


def _factor_admm_system(
    system: Array, preferred_jitter: float | None = None
) -> tuple[tuple[Array, bool], float]:
    """Factor a symmetric ADMM system with explicit numerical safeguards."""
    symmetric = 0.5 * (system + system.T)
    candidates = (1e-10, 1e-8, 1e-6, 1e-4)
    if preferred_jitter is not None:
        candidates = (preferred_jitter,) + tuple(
            value for value in candidates if value != preferred_jitter
        )
    for jitter in candidates:
        try:
            return (
                cho_factor(
                    symmetric + jitter * np.eye(system.shape[0]),
                    lower=True,
                    check_finite=False,
                ),
                jitter,
            )
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("ADMM system is not positive definite after safeguards")


def _project_l1_ball(value: Array, radius: float) -> Array:
    if radius < 0:
        raise ValueError("The fused budget must be nonnegative")
    absolute = np.abs(value)
    if absolute.sum() <= radius:
        return value.copy()
    sorted_values = np.sort(absolute)[::-1]
    cumulative = np.cumsum(sorted_values)
    indices = np.arange(1, value.size + 1)
    valid = sorted_values - (cumulative - radius) / indices > 0
    if not np.any(valid):
        threshold = (cumulative[-1] - radius) / value.size
    else:
        last = np.flatnonzero(valid)[-1]
        threshold = (cumulative[last] - radius) / (last + 1)
    return np.sign(value) * np.maximum(absolute - threshold, 0.0)


def _admm_residuals(
    difference: csr_matrix,
    beta: Array,
    z: Array,
    previous_z: Array,
    rho: float,
) -> tuple[Array, Array]:
    """Return standard ADMM primal and dual residual vectors."""
    primal = np.asarray(difference @ beta).reshape(-1) - z
    dual = rho * np.asarray(difference.T @ (z - previous_z)).reshape(-1)
    return primal, dual


def _enforce_tv_feasibility(
    beta: Array,
    edges: tuple[tuple[int, int], ...],
    fused_budget: float,
    correction_limit: float = 1e-3,
) -> tuple[Array, bool]:
    total_variation = fused_lasso_penalty(beta, edges)
    violation = total_variation - fused_budget
    if violation <= 0.0:
        return np.asarray(beta, dtype=np.float64), False
    if violation > correction_limit:
        raise RuntimeError("LEMNA ADMM returned an infeasible solution")
    if total_variation == 0.0:
        return np.asarray(beta, dtype=np.float64), False
    return np.asarray(beta, dtype=np.float64) * (fused_budget / total_variation), True


def mixture_log_likelihood(
    x: Array,
    y: Array,
    coefficients: Array,
    intercepts: Array,
    scales: Array,
    weights: Array,
) -> float:
    """Evaluate the finite mixture Gaussian-regression log likelihood."""
    means = x @ coefficients.T + intercepts
    residual = (y[:, None] - means) / scales[None, :]
    component = (
        np.log(np.maximum(weights, 1e-300))[None, :]
        - np.log(scales)[None, :]
        - 0.5 * residual**2
        - 0.5 * np.log(2.0 * np.pi)
    )
    return float(logsumexp(component, axis=1).sum())


def generate_mask_neighborhood(
    instance: Array,
    samples: int,
    seed: int,
    mask_probability: float = 0.5,
) -> Array:
    """Generate LEMNA's local binary-mask neighborhood.

    The paper's Section 4.3 neighborhood randomly nullifies a subset of
    features. Row zero is the unmodified instance; remaining rows apply
    independent masks. For EMBER, nullification is a declared feature-space
    adaptation, not a claim about valid PE transformations.
    """
    if samples < 2 or not 0.0 <= mask_probability <= 1.0:
        raise ValueError("Invalid neighborhood size or mask probability")
    value = np.asarray(instance, dtype=np.float64).reshape(-1)
    if not np.isfinite(value).all():
        raise ValueError("Neighborhood seed must be finite")
    rng = np.random.default_rng(seed)
    masks = rng.random((samples - 1, value.size)) >= mask_probability
    neighborhood = np.empty((samples, value.size), dtype=np.float64)
    neighborhood[0] = value
    neighborhood[1:] = masks * value
    return neighborhood


def _weighted_penalized_regression(
    x: Array,
    y: Array,
    responsibility: Array,
    fused_budget: float,
    max_iter: int,
    config_edges: tuple[tuple[int, int], ...] | None,
    optimizer: str = "admm",
    admm_rho: float = 1.0,
    admm_abs_tolerance: float = 1e-6,
    admm_relative_tolerance: float = 1e-5,
    adaptive_rho: bool = False,
    rho_mu: float = 10.0,
    rho_tau: float = 2.0,
    factor_jitter: float | None = None,
    initial_beta: Array | None = None,
    over_relaxation: float = 1.0,
    failure_callback: Callable[[dict], None] | None = None,
) -> tuple[Array, float, OptimizerDiagnostics]:
    """Optimize one weighted fused-lasso regression with an explicit constraint.

    The weighted squared loss without an intercept is minimized subject to the
    exact total-variation bound from Equation (7) of Guo et al. Auxiliary
    variables linearize the absolute-value constraints.
    """
    if optimizer not in {"admm", "slsqp"}:
        raise ValueError("Optimizer must be 'admm' or 'slsqp'")
    if not 1.0 <= over_relaxation < 2.0:
        raise ValueError("ADMM over_relaxation must be in [1, 2)")
    weights = np.asarray(responsibility, dtype=np.float64)
    denominator = max(float(weights.sum()), 1e-12)
    n_features = x.shape[1]
    edges = config_edges
    if edges is None:
        edges = tuple((index, index + 1) for index in range(n_features - 1))
    if optimizer == "admm":
        difference = _difference_matrix(n_features, edges)
        difference_transpose = difference.T.tocsr()
        weighted_x = x * np.sqrt(weights[:, None] / denominator)
        weighted_y = y * np.sqrt(weights / denominator)
        gram = weighted_x.T @ weighted_x
        rho = float(admm_rho)
        if rho <= 0.0 or rho_mu <= 1.0 or rho_tau <= 1.0:
            raise ValueError("ADMM rho and adaptive-rho parameters must be valid")

        def factor_for(current_rho: float):
            return _factor_admm_system(
                gram + current_rho * (difference_transpose @ difference).toarray(),
                factor_jitter,
            )

        factor, factor_jitter = factor_for(rho)
        base_rhs = weighted_x.T @ weighted_y
        beta = (
            np.zeros(n_features)
            if initial_beta is None
            else np.asarray(initial_beta, dtype=np.float64).reshape(-1).copy()
        )
        if beta.shape != (n_features,) or not np.isfinite(beta).all():
            raise ValueError("initial_beta has the wrong shape or non-finite values")
        z = _project_l1_ball(np.asarray(difference @ beta).reshape(-1), fused_budget)
        dual = np.zeros(len(edges))
        objective_history: list[float] = []
        converged = False
        primal_residual = dual_residual = np.inf
        primal_tolerance = dual_tolerance = np.inf
        primal_history: list[float] = []
        dual_history: list[float] = []
        primal_tolerance_history: list[float] = []
        dual_tolerance_history: list[float] = []
        rho_history: list[float] = []
        for iteration in range(1, max_iter + 1):
            rhs = base_rhs + rho * difference_transpose @ (z - dual)
            beta = cho_solve(factor, rhs, check_finite=False)
            difference_beta = np.asarray(difference @ beta).reshape(-1)
            previous_z = z
            relaxed_difference = (
                over_relaxation * difference_beta
                + (1.0 - over_relaxation) * previous_z
            )
            z = _project_l1_ball(relaxed_difference + dual, fused_budget)
            dual += difference_beta - z
            primal_vector, dual_vector = _admm_residuals(
                difference, beta, z, previous_z, rho
            )
            primal_residual = np.linalg.norm(primal_vector)
            dual_residual = np.linalg.norm(dual_vector)
            primal_tolerance = np.sqrt(len(edges)) * admm_abs_tolerance + admm_relative_tolerance * max(
                np.linalg.norm(difference_beta), np.linalg.norm(z)
            )
            dual_tolerance = np.sqrt(n_features) * admm_abs_tolerance + admm_relative_tolerance * np.linalg.norm(
                rho * difference_transpose @ dual
            )
            primal_history.append(float(primal_residual))
            dual_history.append(float(dual_residual))
            primal_tolerance_history.append(float(primal_tolerance))
            dual_tolerance_history.append(float(dual_tolerance))
            rho_history.append(float(rho))
            objective_history.append(
                float(0.5 * np.sum(weights * (x @ beta - y) ** 2) / denominator)
            )
            if primal_residual <= primal_tolerance and dual_residual <= dual_tolerance:
                # Do not accept a residual certificate that would immediately
                # be invalidated by the final TV-feasibility correction.
                raw_total_variation = fused_lasso_penalty(beta, edges)
                if raw_total_variation <= fused_budget:
                    converged = True
                    break
            if adaptive_rho:
                old_rho = rho
                if primal_residual > rho_mu * dual_residual:
                    rho = min(rho * rho_tau, 1e6)
                elif dual_residual > rho_mu * primal_residual:
                    rho = max(rho / rho_tau, 1e-6)
                if rho != old_rho:
                    dual *= old_rho / rho
                    factor, factor_jitter = factor_for(rho)
        fitted = np.asarray(beta, dtype=np.float64)
        fitted, correction_applied = _enforce_tv_feasibility(
            fitted, edges, fused_budget
        )
        total_variation = fused_lasso_penalty(fitted, edges)
        if correction_applied:
            primal_residual = float(np.linalg.norm(difference @ fitted - z))
            converged = (
                primal_residual <= primal_tolerance
                and dual_residual <= dual_tolerance
            )
        loss = float(0.5 * np.sum(weights * (x @ fitted - y) ** 2) / denominator)
        diagnostics = OptimizerDiagnostics(
            optimizer="admm",
            iterations=iteration,
            converged=converged,
            primal_residual=float(primal_residual),
            dual_residual=float(dual_residual),
            primal_tolerance=float(primal_tolerance),
            dual_tolerance=float(dual_tolerance),
            objective_history=tuple(objective_history),
            correction_applied=correction_applied,
            total_variation=float(total_variation),
            termination_status="converged" if converged else "max_iterations",
            factor_jitter=factor_jitter,
            primal_residual_history=tuple(primal_history),
            dual_residual_history=tuple(dual_history),
            primal_tolerance_history=tuple(primal_tolerance_history),
            dual_tolerance_history=tuple(dual_tolerance_history),
            rho_history=tuple(rho_history),
        )
        if failure_callback is not None and not converged:
            failure_callback({
                "x": x.copy(),
                "y": y.copy(),
                "responsibility": weights.copy(),
                "difference": difference.copy(),
                "beta": beta.copy(),
                "z": z.copy(),
                "dual": dual.copy(),
                "rho": rho,
                "diagnostics": diagnostics,
            })
        return fitted, loss, diagnostics
    # A constant coefficient vector is always feasible, even when the
    # unconstrained least-squares solution exceeds the fusion budget.
    initial_parameters = np.concatenate((np.zeros(n_features), np.zeros(len(edges))))

    def objective(parameters: Array) -> float:
        beta = parameters[:n_features]
        residual = x @ beta - y
        return float(
            0.5 * np.sum(weights * residual**2) / denominator
        )

    def objective_gradient(parameters: Array) -> Array:
        beta = parameters[:n_features]
        residual = x @ beta - y
        gradient = np.zeros_like(parameters)
        gradient[:n_features] = (x.T @ (weights * residual)) / denominator
        return gradient

    constraint_matrix = lil_matrix((2 * len(edges) + 1, n_features + len(edges)))
    for edge_index, (left, right) in enumerate(edges):
        constraint_matrix[edge_index, n_features + edge_index] = 1.0
        constraint_matrix[edge_index, left] = 1.0
        constraint_matrix[edge_index, right] = -1.0
        constraint_matrix[len(edges) + edge_index, n_features + edge_index] = 1.0
        constraint_matrix[len(edges) + edge_index, left] = -1.0
        constraint_matrix[len(edges) + edge_index, right] = 1.0
    constraint_matrix[-1, n_features:] = -1.0
    linear_constraint = LinearConstraint(
        constraint_matrix.tocsr(),
        np.concatenate((np.zeros(2 * len(edges)), [-fused_budget])),
        np.concatenate((np.full(2 * len(edges), np.inf), [np.inf])),
    )

    result = minimize(
        objective,
        initial_parameters,
        jac=objective_gradient,
        method="SLSQP",
        constraints=(linear_constraint,),
        options={"maxiter": max_iter, "ftol": 1e-9, "disp": False},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"LEMNA fused-lasso optimization failed: {result.message}")
    fitted = np.asarray(result.x[:n_features], dtype=np.float64)
    if fused_lasso_penalty(fitted, edges) > fused_budget + 1e-7:
        raise RuntimeError("LEMNA fused-lasso optimizer returned an infeasible solution")
    diagnostics = OptimizerDiagnostics(
        optimizer="slsqp",
        iterations=int(getattr(result, "nit", max_iter)),
        converged=bool(result.success),
        primal_residual=0.0,
        dual_residual=0.0,
        primal_tolerance=0.0,
        dual_tolerance=0.0,
        objective_history=(float(result.fun),),
        correction_applied=False,
        total_variation=fused_lasso_penalty(fitted, edges),
        termination_status="converged" if result.success else "solver_failure",
        factor_jitter=0.0,
    )
    return fitted, float(result.fun), diagnostics


def select_component(
    target_output: float,
    target_means: Array,
    scales: Array,
    weights: Array,
) -> int:
    """Select the component with greatest target-output log likelihood.

    Section 4.3 states that the component best approximating the target local
    decision boundary is selected, but does not define a numerical score.
    This function is the declared maximum component log-likelihood
    operationalization, not a claim of verbatim paper reproduction.
    """
    scores = (
        np.log(np.maximum(weights, 1e-300))
        - np.log(np.maximum(scales, 1e-300))
        - 0.5 * ((float(target_output) - target_means) / scales) ** 2
    )
    return int(np.argmax(scores))


def fit_lemna(
    neighborhood: Array,
    outputs: Array,
    config: LemnaConfig | None = None,
    target: Array | None = None,
    target_output: float | None = None,
    failure_callback: Callable[[dict], None] | None = None,
) -> LemnaResult:
    """Fit LEMNA and return coefficient-derived attributions.

    `neighborhood` is the local design matrix and `outputs` are black-box
    scalar outputs in the same row order. For an explanation target, pass the
    original instance as `target`; its highest-responsibility mixture component
    supplies the local attribution vector from the selected component's
    coefficients, as described in Section 4.3 of the paper.
    """
    cfg = config or LemnaConfig()
    x = np.asarray(neighborhood, dtype=np.float64)
    y = np.asarray(outputs, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or y.shape[0] != x.shape[0]:
        raise ValueError("Neighborhood and output dimensions do not agree")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("LEMNA inputs must be finite")
    n_rows, n_features = x.shape
    if n_rows <= cfg.components or n_features == 0:
        raise ValueError("LEMNA needs more rows than mixture components")
    rng = np.random.default_rng(cfg.seed)
    coefficients = rng.normal(0.0, 1e-3, (cfg.components, n_features))
    intercepts = np.zeros(cfg.components)
    scales = np.full(cfg.components, max(float(y.std()), cfg.min_scale))
    weights = np.full(cfg.components, 1.0 / cfg.components)
    previous = -np.inf
    converged = False
    optimizer_diagnostics: list[OptimizerDiagnostics] = []
    em_trace: list[EMIterationDiagnostics] = []
    for iteration in range(1, cfg.max_iter + 1):
        means = x @ coefficients.T
        standardized = (y[:, None] - means) / scales[None, :]
        log_prob = (
            np.log(np.maximum(weights, 1e-300))[None, :]
            - np.log(scales)[None, :]
            - 0.5 * standardized**2
            - 0.5 * np.log(2.0 * np.pi)
        )
        responsibilities = np.exp(log_prob - logsumexp(log_prob, axis=1, keepdims=True))
        weights = np.maximum(responsibilities.mean(axis=0), 1e-8)
        weights /= weights.sum()
        previous_coefficients = coefficients.copy()
        iteration_diagnostics: list[OptimizerDiagnostics] = []
        for component in range(cfg.components):
            fit, _, diagnostics = _weighted_penalized_regression(
                x,
                y,
                responsibilities[:, component],
                cfg.fused_budget,
                cfg.inner_max_iter,
                cfg.fusion_edges,
                cfg.optimizer,
                cfg.admm_rho,
                cfg.admm_abs_tolerance,
                cfg.admm_relative_tolerance,
                cfg.admm_adaptive_rho,
                cfg.admm_rho_mu,
                cfg.admm_rho_tau,
                cfg.admm_factor_jitter,
                failure_callback=failure_callback,
            )
            optimizer_diagnostics.append(diagnostics)
            iteration_diagnostics.append(diagnostics)
            coefficients[component] = fit
            residual = y - x @ coefficients[component]
            scales[component] = max(
                cfg.min_scale,
                float(np.sqrt(np.sum(responsibilities[:, component] * residual**2)
                             / max(responsibilities[:, component].sum(), 1e-12))),
            )
        likelihood = mixture_log_likelihood(
            x, y, coefficients, intercepts, scales, weights
        )
        absolute_change = (
            float("inf") if not np.isfinite(previous)
            else abs(likelihood - previous)
        )
        relative_change = (
            float("inf") if not np.isfinite(previous)
            else absolute_change / max(1.0, abs(previous))
        )
        em_trace.append(
            EMIterationDiagnostics(
                iteration=iteration,
                log_likelihood=float(likelihood),
                absolute_likelihood_change=float(absolute_change),
                relative_likelihood_change=float(relative_change),
                weights=tuple(float(value) for value in weights),
                scales=tuple(float(value) for value in scales),
                coefficient_changes=tuple(
                    float(np.linalg.norm(coefficients[index] - previous_coefficients[index]))
                    for index in range(cfg.components)
                ),
                inner_converged=tuple(item.converged for item in iteration_diagnostics),
                inner_iterations=tuple(item.iterations for item in iteration_diagnostics),
                inner_primal_residuals=tuple(item.primal_residual for item in iteration_diagnostics),
                inner_dual_residuals=tuple(item.dual_residual for item in iteration_diagnostics),
                inner_primal_tolerances=tuple(item.primal_tolerance for item in iteration_diagnostics),
                inner_dual_tolerances=tuple(item.dual_tolerance for item in iteration_diagnostics),
                inner_factor_jitters=tuple(item.factor_jitter for item in iteration_diagnostics),
                inner_correction_applied=tuple(
                    item.correction_applied for item in iteration_diagnostics
                ),
                inner_total_variations=tuple(
                    item.total_variation for item in iteration_diagnostics
                ),
            )
        )
        if absolute_change <= cfg.tolerance * (1.0 + abs(likelihood)):
            converged = all(item.converged for item in iteration_diagnostics)
            break
        previous = likelihood

    if target is None:
        target = x[0]
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if target.shape[0] != n_features:
        raise ValueError("Target dimension does not match neighborhood")
    if target_output is None:
        target_output = float(y[0])
    target_means = target @ coefficients.T
    selected = select_component(float(target_output), target_means, scales, weights)
    attribution = coefficients[selected].copy()
    objective = -mixture_log_likelihood(x, y, coefficients, intercepts, scales, weights)
    return LemnaResult(
        coefficients=coefficients,
        intercepts=intercepts,
        scales=scales,
        weights=weights,
        responsibilities=responsibilities,
        log_likelihood=mixture_log_likelihood(
            x, y, coefficients, intercepts, scales, weights
        ),
        objective=float(objective),
        attribution=attribution,
        selected_component=selected,
        iterations=iteration,
        converged=converged,
        optimizer_diagnostics=tuple(optimizer_diagnostics),
        em_trace=tuple(em_trace),
    )


def explain_black_box(
    predict: Callable[[Array], Array],
    instance: Array,
    neighborhood: Array | None = None,
    config: LemnaConfig | None = None,
) -> LemnaResult:
    """Fit binary-class LEMNA from a scalar target-class predictor.

    The binary-class approximation follows Section 4.3: one scalar mixture
    regression is fitted to the selected class probability. Multiclass
    vector-output approximation is intentionally not implemented.
    """
    cfg = config or LemnaConfig()
    if neighborhood is None:
        neighborhood = generate_mask_neighborhood(
            instance, cfg.neighborhood_samples, cfg.seed, cfg.mask_probability
        )
    x = np.asarray(neighborhood, dtype=np.float64)
    prediction = np.asarray(predict(x), dtype=np.float64)
    if prediction.shape not in {(x.shape[0],), (x.shape[0], 1)}:
        raise ValueError("Predictor must return one scalar output per batch row")
    y = prediction.reshape(-1)
    instance_value = np.asarray(instance, dtype=np.float64).reshape(-1)
    instance_prediction = np.asarray(predict(instance_value[None, :]), dtype=np.float64)
    if instance_prediction.size != 1:
        raise ValueError("Predictor must return one scalar for one instance")
    return fit_lemna(
        x, y, config=cfg, target=instance_value,
        target_output=float(instance_prediction.reshape(-1)[0]),
    )
