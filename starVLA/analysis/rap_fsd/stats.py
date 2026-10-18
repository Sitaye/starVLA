# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""RAP-FSD Stage A component 3 core statistics (pure numpy, CPU, fully unit-testable).

Every public function cites the protocol clause it implements
(docs/research/ac_audit_rapfsd_math_20261018.md section 8, pre-registered protocol v1.2,
thresholds frozen 2026-10-18; method definitions from docs/research/new_idea_10_18.md
final section).

Design rules (protocol section 8 / audit section 2):
* sklearn is NOT a repo dependency (requirements.txt) - ridge (analytic solve) and CCA
  (SVD of the whitened cross-covariance) are hand-written here;
* all estimators are episode-pool level: rows are transition-level but every significance
  statement is episode-level (episode-block permutation / episode-block bootstrap);
  frame-level p-values are forbidden (audit 2.3 autocorrelation / oracle H3);
* everything is conservative by default: max-over-k null distribution, ridge-regularised
  CCA, OOF conditional means, no selection on the holdout pool;
* all randomness goes through explicit seeded ``numpy.random.default_rng`` (seed=42 global).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SEED_DEFAULT = 42

# protocol section 8 (v1.2, frozen): CCA ridge starts at 0.1*I (on unit-variance
# standardised covariances) with a lambda sensitivity curve; delta-R2 gate threshold 1%
# of Var(A_tilde); Freedman-Lane residual permutation >= 1000 reps, max-over-k null;
# episode-level block bootstrap 95% CI; predictability probe >= 50% of Var(z_{1:k*})
# (linear + small MLP); fold-consistency median principal angle < 45 degrees.
CCA_RIDGE_LAMBDA = 0.1
CCA_LAMBDA_GRID = (0.01, 0.05, 0.1, 0.5, 1.0)
PROBE_RIDGE_LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
K_CANDIDATES = (4, 8, 16, 32)  # protocol section 8: k in {4, 8, 16, 32} + "do not train"
DELTA_R2_THRESHOLD = 0.01  # G2: delta R2(k*) >= 1% of Var(A_tilde)
PROBE_THRESHOLD = 0.50  # G3: >= 50% of Var(z_{1:k*}) explained
PRINCIPAL_ANGLE_THRESHOLD_DEG = 45.0  # G4: median < 45 degrees
N_PERMS_MIN = 1000  # G1: Freedman-Lane >= 1000
N_BOOTSTRAP_DEFAULT = 1000
CV_SPLITS_DEFAULT = 5


# ---------------------------------------------------------------------------
# ridge regression (analytic solve; no sklearn)
# ---------------------------------------------------------------------------


class RidgeRegressor:
    """Multi-output ridge with per-column standardisation and centring.

    X is z-scored with the fit-pool column means/stds (zero-std columns pass through),
    the closed-form solve is (X^T X + lam*I) W = X^T Y (gram is positive definite for
    lam > 0, so a plain solve suffices).
    """

    def __init__(self, lam: float):
        self.lam = float(lam)
        self.mean_x = None
        self.std_x = None
        self.mean_y = None
        self.coef = None  # [d_x, d_y] in standardised-X coordinates

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mean_x = x.mean(axis=0)
        std_x = x.std(axis=0)
        self.std_x = np.where(std_x > 1e-12, std_x, 1.0)
        self.mean_y = y.mean(axis=0)
        xs = (x - self.mean_x) / self.std_x
        ys = y - self.mean_y
        gram = xs.T @ xs
        d = gram.shape[0]
        gram = gram + self.lam * np.eye(d)
        rhs = xs.T @ ys
        self.coef = np.linalg.solve(gram, rhs)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        xs = (np.asarray(x, dtype=np.float64) - self.mean_x) / self.std_x
        return xs @ self.coef + self.mean_y


def kfold_indices(n: int, n_splits: int, seed: int = SEED_DEFAULT) -> list[np.ndarray]:
    """Deterministic K-fold index assignment (seeded shuffle of row order)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return [np.sort(chunk) for chunk in np.array_split(order, max(2, n_splits))]


def select_lambda_ridge(
    x: np.ndarray,
    y: np.ndarray,
    lambdas: tuple[float, ...] = PROBE_RIDGE_LAMBDA_GRID,
    n_splits: int = CV_SPLITS_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> tuple[float, dict]:
    """Fold-internal K-fold CV lambda selection (protocol section 8: lambda via fold-internal K-fold CV)."""
    n = x.shape[0]
    folds = kfold_indices(n, min(n_splits, max(2, n)), seed=seed)
    curve = {}
    for lam in lambdas:
        sq_err = 0.0
        for val_idx in folds:
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_idx] = False
            model = RidgeRegressor(lam).fit(x[train_mask], y[train_mask])
            pred = model.predict(x[val_idx])
            sq_err += float(((y[val_idx] - pred) ** 2).sum())
        curve[float(lam)] = sq_err / n
    best = min(curve, key=curve.get)
    return best, curve


def oof_ridge_predict(
    x: np.ndarray,
    y: np.ndarray,
    lam: float,
    n_splits: int = CV_SPLITS_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> np.ndarray:
    """K-fold out-of-fold ridge predictions on the given pool (no self-fit leakage)."""
    n = x.shape[0]
    folds = kfold_indices(n, min(n_splits, max(2, n)), seed=seed)
    pred = np.empty_like(np.asarray(y, dtype=np.float64))
    for val_idx in folds:
        train_mask = np.ones(n, dtype=bool)
        train_mask[val_idx] = False
        model = RidgeRegressor(lam).fit(x[train_mask], y[train_mask])
        pred[val_idx] = model.predict(x[val_idx])
    return pred


# ---------------------------------------------------------------------------
# conditional residualisation (protocol section 8 fit; new_idea section 3 FWL step)
# ---------------------------------------------------------------------------


@dataclass
class ConditionalMean:
    """mu-hat(C) ridge model with CV-selected lambda, stored with its lambda for reproducibility."""

    model: RidgeRegressor
    lam: float

    def predict(self, c: np.ndarray) -> np.ndarray:
        return self.model.predict(c)


def fit_conditional_mean(
    c: np.ndarray,
    y: np.ndarray,
    lambdas: tuple[float, ...] = PROBE_RIDGE_LAMBDA_GRID,
    n_splits: int = CV_SPLITS_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> ConditionalMean:
    """Fit mu-hat_A(C) / mu-hat_F(C): ridge with lambda chosen by fold-internal K-fold CV."""
    lam, _ = select_lambda_ridge(c, y, lambdas=lambdas, n_splits=n_splits, seed=seed)
    model = RidgeRegressor(lam).fit(c, y)
    return ConditionalMean(model=model, lam=lam)


def residualize(y: np.ndarray, c: np.ndarray, mu: ConditionalMean) -> np.ndarray:
    """y_tilde = y - mu-hat(C) (new_idea section 3: A_tilde, F_tilde)."""
    return np.asarray(y, dtype=np.float64) - mu.predict(np.asarray(c, dtype=np.float64))


# ---------------------------------------------------------------------------
# CCA on whitened cross-covariance (new_idea section 3; protocol section 8: ridge 0.1*I + lambda curve)
# ---------------------------------------------------------------------------


def _matrix_inv_sqrt(matrix: np.ndarray, jitter: float = 1e-10) -> np.ndarray:
    """Symmetric inverse square root via eigh with eigenvalue floor (Sigma_F^{-1/2})."""
    matrix = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    scale = float(np.max(np.abs(eigvals), initial=1.0))
    floor = max(jitter * scale, 1e-12)
    eigvals = np.maximum(eigvals, floor)
    return (eigvecs / np.sqrt(eigvals)) @ eigvecs.T


@dataclass
class CCAProjection:
    """Whitened CCA: z_n = U_k^T Sigma_F^{-1/2} F_tilde_n (new_idea section 4 boxed definition).

    Because Sigma_F^{-1/2} is symmetric, ``U_k^T Sigma_F^{-1/2}`` equals
    ``(Sigma_F^{-1/2} U_k)^T``, i.e. z is the projection of standardised F_tilde onto the
    flow-side loadings ``a_flow``. Variates have unit variance by construction
    (Var(a_j^T F_tilde_std) = 1), matching the "z whitened to unit variance" contract
    used by the Stage B loss normalisation (audit 2.4 M2).
    """

    mean_f: np.ndarray
    std_f: np.ndarray
    mean_a: np.ndarray
    std_a: np.ndarray
    a_flow: np.ndarray  # [d_F, r] flow-side canonical loadings (standardised F_tilde space)
    b_action: np.ndarray  # [d_A, r] action-side canonical loadings
    rho: np.ndarray  # [r] canonical correlations, descending
    ridge_lambda: float

    def project(self, f_tilde: np.ndarray, k: int) -> np.ndarray:
        f_std = (np.asarray(f_tilde, dtype=np.float64) - self.mean_f) / self.std_f
        return f_std @ self.a_flow[:, :k]


def fit_cca(
    f_tilde: np.ndarray,
    a_tilde: np.ndarray,
    ridge_lambda: float = CCA_RIDGE_LAMBDA,
) -> CCAProjection:
    """M = Sigma_F^{-1/2} Sigma_FA Sigma_A^{-1/2}; SVD M = U diag(rho) V^T;
    a_flow = Sigma_F^{-1/2} U, b_action = Sigma_A^{-1/2} V.

    Inputs are z-scored with the fit-pool stats so Sigma has unit diagonal and the frozen
    ``0.1*I`` ridge is applied literally (protocol section 8 fit).
    """
    f = np.asarray(f_tilde, dtype=np.float64)
    a = np.asarray(a_tilde, dtype=np.float64)
    n = f.shape[0]
    mean_f, std_f = f.mean(axis=0), f.std(axis=0)
    mean_a, std_a = a.mean(axis=0), a.std(axis=0)
    std_f = np.where(std_f > 1e-12, std_f, 1.0)
    std_a = np.where(std_a > 1e-12, std_a, 1.0)
    fs = (f - mean_f) / std_f
    as_ = (a - mean_a) / std_a
    d_f, d_a = fs.shape[1], as_.shape[1]
    sigma_f = fs.T @ fs / n + ridge_lambda * np.eye(d_f)
    sigma_a = as_.T @ as_ / n + ridge_lambda * np.eye(d_a)
    sigma_fa = fs.T @ as_ / n
    inv_sqrt_f = _matrix_inv_sqrt(sigma_f)
    inv_sqrt_a = _matrix_inv_sqrt(sigma_a)
    m = inv_sqrt_f @ sigma_fa @ inv_sqrt_a
    u, singulars, vt = np.linalg.svd(m, full_matrices=False)
    return CCAProjection(
        mean_f=mean_f,
        std_f=std_f,
        mean_a=mean_a,
        std_a=std_a,
        a_flow=inv_sqrt_f @ u,
        b_action=inv_sqrt_a @ vt.T,
        rho=singulars,
        ridge_lambda=float(ridge_lambda),
    )


# ---------------------------------------------------------------------------
# delta R2(k) (new_idea section 5 definition; protocol section 8 G2 threshold frozen at 1%)
# ---------------------------------------------------------------------------


def r2_explained(a_tilde: np.ndarray, a_hat: np.ndarray) -> float:
    """delta R2 = 1 - E||A_tilde - a_hat||^2 / E||A_tilde||^2 where E is the row mean of
    squared norms, WITHOUT re-centering: A_tilde is already conditionally residualised,
    so the literal new_idea section 5 formula applies."""
    a_tilde = np.asarray(a_tilde, dtype=np.float64)
    resid = a_tilde - np.asarray(a_hat, dtype=np.float64)
    total = float((a_tilde**2).sum())
    if total <= 0:
        return 0.0
    return 1.0 - float((resid**2).sum()) / total


def _fit_linear_probe(z: np.ndarray, a_tilde: np.ndarray, lam: float) -> np.ndarray:
    """Return the coefficient matrix mapping z to A_tilde (near-OLS when lam is tiny)."""
    z = np.asarray(z, dtype=np.float64)
    gram = z.T @ z + lam * np.eye(z.shape[1])
    return np.linalg.solve(gram, z.T @ np.asarray(a_tilde, dtype=np.float64))


def fit_linear_probe(z: np.ndarray, a_tilde: np.ndarray, lam: float) -> np.ndarray:
    """Public wrapper: fit the linear probe (coefficient matrix) mapping z to A_tilde."""
    return _fit_linear_probe(z, a_tilde, lam)


def delta_r2_in_sample_curve(
    z: np.ndarray,
    a_tilde: np.ndarray,
    k_candidates: tuple[int, ...] = K_CANDIDATES,
) -> dict[int, float]:
    """In-sample delta R2(k) with near-OLS probes (lam = 1e-8).

    Nested column spaces + least squares imply RSS non-increasing in k, hence the curve
    is monotone non-decreasing (used for the monotonicity unit test and descriptive
    curves only - gate statistics always use OOF / holdout evaluation).
    """
    z = np.asarray(z, dtype=np.float64)
    a_tilde = np.asarray(a_tilde, dtype=np.float64)
    curve = {}
    for k in k_candidates:
        if k > z.shape[1]:
            continue
        coef = _fit_linear_probe(z[:, :k], a_tilde, lam=1e-8)
        curve[int(k)] = r2_explained(a_tilde, z[:, :k] @ coef)
    return curve


def delta_r2_oof_curve(
    z: np.ndarray,
    a_tilde: np.ndarray,
    k_candidates: tuple[int, ...] = K_CANDIDATES,
    lam: float = 1e-2,
    n_splits: int = CV_SPLITS_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> dict[int, float]:
    """OOF delta R2(k) inside one pool: linear probes with K-fold OOF predictions
    (protocol section 8 primary gate statistic)."""
    z = np.asarray(z, dtype=np.float64)
    a_tilde = np.asarray(a_tilde, dtype=np.float64)
    curve = {}
    for k in k_candidates:
        if k > z.shape[1]:
            continue
        pred = oof_ridge_predict(z[:, :k], a_tilde, lam=lam, n_splits=n_splits, seed=seed)
        curve[int(k)] = r2_explained(a_tilde, pred)
    return curve


# ---------------------------------------------------------------------------
# RRR control (audit 2.3: CCA descriptive + RRR predictive, both pre-registered)
# ---------------------------------------------------------------------------


def fit_rrr(f_tilde: np.ndarray, a_tilde: np.ndarray, lam: float = 0.1) -> dict:
    """Reduced-rank ridge: ridge solution beta-hat of F_tilde -> A_tilde, then its SVD
    gives the rank ladder (rank-k predictor = truncated SVD reconstruction)."""
    f = np.asarray(f_tilde, dtype=np.float64)
    a = np.asarray(a_tilde, dtype=np.float64)
    n = f.shape[0]
    gram = f.T @ f / n + lam * np.eye(f.shape[1])
    beta = np.linalg.solve(gram, f.T @ a / n)
    u, s, vt = np.linalg.svd(beta, full_matrices=False)
    return {"beta": beta, "u": u, "s": s, "vt": vt}


# ---------------------------------------------------------------------------
# Freedman-Lane residual permutation (protocol section 8 G1 / oracle H1 + M4)
# ---------------------------------------------------------------------------


def episode_block_permutation(episode_ids: np.ndarray, seed: int) -> np.ndarray:
    """Row gather-index vector realising an EPISODE-level permutation of residuals.

    Episodes are permuted wholesale (audit 2.3: "the whole-episode (F_tilde, A_tilde)
    pair is remapped onto another episode"), preserving within-episode autocorrelation;
    rows of a target episode draw their residual from the mapped source episode at the
    matching fractional position (episode lengths differ under stride subsampling and
    NaN tail removal).
    """
    rng = np.random.default_rng(seed)
    episode_ids = np.asarray(episode_ids)
    unique_eps = np.unique(episode_ids)
    perm_eps = rng.permutation(unique_eps)
    source_of_episode = dict(zip(unique_eps.tolist(), perm_eps.tolist(), strict=True))
    rows_of_episode = {ep: [] for ep in unique_eps.tolist()}
    for row, ep in enumerate(episode_ids.tolist()):
        rows_of_episode[ep].append(row)
    gather = np.empty(episode_ids.shape[0], dtype=np.int64)
    for target_ep, target_rows in rows_of_episode.items():
        source_rows = rows_of_episode[source_of_episode[target_ep]]
        m_t, m_s = len(target_rows), len(source_rows)
        if m_s == 1:
            gather[target_rows] = source_rows[0]
        else:
            positions = np.round(np.arange(m_t) * (m_s - 1) / max(m_t - 1, 1)).astype(np.int64)
            gather[target_rows] = np.asarray(source_rows, dtype=np.int64)[positions]
    return gather


@dataclass
class FreedmanLaneState:
    """Fitted-once FL ingredients: the C-only predictor of A_tilde, split into the fitted
    part and the residual part (Freedman-Lane 1989 construction)."""

    fitted_part: np.ndarray  # P = C-only fit of A_tilde; fixed under permutation
    residual: np.ndarray  # r = A_tilde - P; the only part that is permuted
    gram_cholesky: np.ndarray  # Cholesky factor of C_std^T C_std + lam*I (fixed across perms)
    mean_c: np.ndarray
    std_c: np.ndarray
    mean_a: np.ndarray
    lam: float


def freedman_lane_prepare(a_tilde: np.ndarray, c: np.ndarray, lam: float = 1e-2) -> FreedmanLaneState:
    """Fit the C-only predictor of A_tilde once (Freedman-Lane 1989 construction).

    Permuting only the residual keeps the C-explainable structure exactly invariant
    (verified by test_freedman_lane_preserves_c_structure), so the null varies only in
    the component the gate statistic can attribute to F_tilde.
    """
    a_tilde = np.asarray(a_tilde, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    model = RidgeRegressor(lam).fit(c, a_tilde)
    fitted_part = model.predict(c)
    residual = a_tilde - fitted_part
    cs = (c - model.mean_x) / model.std_x
    gram = cs.T @ cs + lam * np.eye(cs.shape[1])
    return FreedmanLaneState(
        fitted_part=fitted_part,
        residual=residual,
        gram_cholesky=np.linalg.cholesky(gram),
        mean_c=model.mean_x,
        std_c=model.std_x,
        mean_a=a_tilde.mean(axis=0),
        lam=float(lam),
    )


def freedman_lane_null(
    f_tilde: np.ndarray,
    a_tilde: np.ndarray,
    c: np.ndarray,
    episode_ids: np.ndarray,
    k_candidates: tuple[int, ...] = K_CANDIDATES,
    n_perms: int = N_PERMS_MIN,
    cca_ridge_lambda: float = CCA_RIDGE_LAMBDA,
    fl_lam: float = 1e-2,
    probe_lam: float = 1e-2,
    n_splits: int = CV_SPLITS_DEFAULT,
    seed: int = SEED_DEFAULT,
    max_f: int = 56,
) -> dict:
    """Permutation null of the max-over-k OOF delta R2 statistic under H0: no F_tilde -
    A_tilde association beyond C.

    Per permutation (protocol section 8 G1 / audit 2.2: the whole pipeline is refit on
    the permuted data - mu-hat_A, CCA, probes - while F_tilde and its conditional mean
    stay fixed because C is never permuted):
      1. r_perm: episode-block permutation of the FL residual r = A_tilde - C-only fit;
      2. A_perm = C-only fit + r_perm (C-explainable structure preserved exactly);
      3. re-residualise: mu-hat_A refit on (C, A_perm) -> A_perm_res;
      4. CCA refit on (F_tilde, A_perm_res) -> z_perm;
      5. OOF linear-probe delta R2(k) for every k in the candidate set;
      6. record max-over-k (oracle M4: multivariate selection over k x views x method).
    Returns the null array of max-over-k statistics plus per-k nulls for reporting.
    """
    f_tilde = np.asarray(f_tilde, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    a_tilde = np.asarray(a_tilde, dtype=np.float64)
    state = freedman_lane_prepare(a_tilde, c, lam=fl_lam)
    n, d_a = a_tilde.shape
    ks = [k for k in k_candidates if k <= max_f]
    null_max = np.empty(n_perms, dtype=np.float64)
    null_per_k = {int(k): np.empty(n_perms, dtype=np.float64) for k in ks}
    # fixed pieces: F_tilde standardisation and Sigma_F never change under A permutations
    mean_f, std_f = f_tilde.mean(axis=0), f_tilde.std(axis=0)
    std_f = np.where(std_f > 1e-12, std_f, 1.0)
    fs = (f_tilde - mean_f) / std_f
    sigma_f = fs.T @ fs / n + cca_ridge_lambda * np.eye(f_tilde.shape[1])
    inv_sqrt_f = _matrix_inv_sqrt(sigma_f)
    cs = (c - state.mean_c) / state.std_c
    # mean of A_tilde: mu-hat_A on permuted data has the same intercept expectation
    a_mean = state.mean_a
    for perm in range(n_perms):
        perm_seed = seed + 1 + perm  # independent, reproducible stream per permutation
        gather = episode_block_permutation(episode_ids, seed=perm_seed)
        a_perm = state.fitted_part + state.residual[gather]
        # refit mu-hat_A on permuted data via the pre-factorised Gram system (fixed lambda)
        rhs = cs.T @ a_perm
        cho = state.gram_cholesky
        coef_perm = np.linalg.solve(cho.T, np.linalg.solve(cho, rhs))
        a_res_perm = a_perm - (cs @ coef_perm + a_mean)
        # CCA refit on (F_tilde, A_perm_res)
        std_a = a_res_perm.std(axis=0)
        std_a = np.where(std_a > 1e-12, std_a, 1.0)
        as_ = (a_res_perm - a_res_perm.mean(axis=0)) / std_a
        sigma_a = as_.T @ as_ / n + cca_ridge_lambda * np.eye(d_a)
        sigma_fa = fs.T @ as_ / n
        m = inv_sqrt_f @ sigma_fa @ _matrix_inv_sqrt(sigma_a)
        u, _, _ = np.linalg.svd(m, full_matrices=False)
        a_flow = inv_sqrt_f @ u
        z_perm = fs @ a_flow
        perm_curve = delta_r2_oof_curve(
            z_perm, a_res_perm, k_candidates=tuple(ks), lam=probe_lam, n_splits=n_splits, seed=perm_seed
        )
        null_max[perm] = max(perm_curve.values())
        for k in ks:
            null_per_k[int(k)][perm] = perm_curve[int(k)]
    return {"max_over_k": null_max, "per_k": null_per_k}


# ---------------------------------------------------------------------------
# episode-level block bootstrap (protocol section 8 G2: 95% CI lower bound > 0)
# ---------------------------------------------------------------------------


def _group_rows(episode_ids: np.ndarray) -> dict:
    groups: dict = {}
    for row, ep in enumerate(np.asarray(episode_ids).tolist()):
        groups.setdefault(ep, []).append(row)
    return groups


def block_bootstrap_delta_r2(
    a_tilde: np.ndarray,
    a_hat: np.ndarray,
    episode_ids: np.ndarray,
    n_reps: int = N_BOOTSTRAP_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> dict:
    """Bootstrap the pooled delta R2 statistic by resampling EPISODES with replacement.

    Probes are held fixed (they are fitted on the fold pool; resampling the holdout
    episodes quantifies the holdout-sampling uncertainty of the transferred statistic -
    protocol section 8 G2: block-bootstrap 95% CI lower bound > 0).
    Returns the point estimate and the percentile CI.
    """
    a_tilde = np.asarray(a_tilde, dtype=np.float64)
    a_hat = np.asarray(a_hat, dtype=np.float64)
    episode_ids = np.asarray(episode_ids)
    unique_eps = np.unique(episode_ids)
    rows_of_episode = {ep: np.asarray(rows) for ep, rows in _group_rows(episode_ids).items()}
    point = r2_explained(a_tilde, a_hat)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_reps, dtype=np.float64)
    for rep in range(n_reps):
        sampled = rng.choice(unique_eps, size=unique_eps.shape[0], replace=True)
        row_idx = np.concatenate([rows_of_episode[ep] for ep in sampled.tolist()])
        stats[rep] = r2_explained(a_tilde[row_idx], a_hat[row_idx])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi), "stats": stats}


# ---------------------------------------------------------------------------
# predictability probes (protocol section 8 G3: linear + small MLP, >= 50% of Var(z))
# ---------------------------------------------------------------------------


def linear_probe_r2(
    probe_x_fit: np.ndarray,
    z_fit: np.ndarray,
    probe_x_eval: np.ndarray,
    z_eval: np.ndarray,
    lambdas: tuple[float, ...] = PROBE_RIDGE_LAMBDA_GRID,
    n_splits: int = CV_SPLITS_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> dict:
    """Linear (ridge) probe of z from frozen features; returns eval-pool explained R2."""
    lam, curve = select_lambda_ridge(probe_x_fit, z_fit, lambdas=lambdas, n_splits=n_splits, seed=seed)
    model = RidgeRegressor(lam).fit(probe_x_fit, z_fit)
    return {
        "r2": r2_explained(z_eval, model.predict(probe_x_eval)),
        "lam": lam,
        "cv_curve": curve,
    }


def mlp_probe_r2(
    probe_x_fit: np.ndarray,
    z_fit: np.ndarray,
    probe_x_eval: np.ndarray,
    z_eval: np.ndarray,
    hidden: int = 256,
    seed: int = SEED_DEFAULT,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_frac: float = 0.1,
    patience: int = 20,
) -> dict:
    """Small numpy MLP probe (one hidden layer, ReLU, Adam, early stopping on a seeded
    internal validation split).

    oracle L1: a linear-only probe underestimates learnability and could falsely kill a
    learnable target - hence the frozen two-tier (linear + small MLP) protocol clause.
    """
    rng = np.random.default_rng(seed)
    probe_x_fit = np.asarray(probe_x_fit, dtype=np.float64)
    z_fit = np.asarray(z_fit, dtype=np.float64)
    probe_x_eval = np.asarray(probe_x_eval, dtype=np.float64)
    z_eval = np.asarray(z_eval, dtype=np.float64)
    mean_x, std_x = probe_x_fit.mean(axis=0), probe_x_fit.std(axis=0)
    std_x = np.where(std_x > 1e-12, std_x, 1.0)
    x_all = (probe_x_fit - mean_x) / std_x
    x_eval = (probe_x_eval - mean_x) / std_x
    n, d_in = x_all.shape
    d_out = z_fit.shape[1]
    val_n = max(1, int(n * val_frac))
    perm = rng.permutation(n)
    val_idx, train_idx = perm[:val_n], perm[val_n:]
    w1 = rng.normal(0.0, np.sqrt(2.0 / d_in), size=(d_in, hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0.0, np.sqrt(2.0 / hidden), size=(hidden, d_out))
    b2 = np.zeros(d_out)
    params = [w1, b1, w2, b2]
    moments = [[np.zeros_like(p), np.zeros_like(p)] for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    best_val, best_params, wait = np.inf, None, 0
    epochs_run = 0
    step = 0
    while epochs_run < epochs:
        epochs_run += 1
        order = rng.permutation(train_idx.shape[0])
        for start in range(0, order.shape[0], batch_size):
            step += 1
            batch = train_idx[order[start : start + batch_size]]
            x, target = x_all[batch], z_fit[batch]
            h = np.maximum(x @ w1 + b1, 0.0)
            out = h @ w2 + b2
            diff = out - target
            d_out_d = 2.0 * diff / diff.size
            grads = [
                x.T @ (d_out_d @ w2.T),
                (d_out_d @ w2.T).sum(axis=0),
                h.T @ d_out_d,
                d_out_d.sum(axis=0),
            ]
            for p_idx, (param, grad) in enumerate(zip(params, grads, strict=True)):
                m_v, v_v = moments[p_idx]
                m_v = beta1 * m_v + (1 - beta1) * grad
                v_v = beta2 * v_v + (1 - beta2) * grad * grad
                moments[p_idx] = [m_v, v_v]
                m_hat = m_v / (1 - beta1**step)
                v_hat = v_v / (1 - beta2**step)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps) + lr * weight_decay * param
        h_val = np.maximum(x_all[val_idx] @ w1 + b1, 0.0)
        val_loss = float(((z_fit[val_idx] - (h_val @ w2 + b2)) ** 2).mean())
        if val_loss < best_val - 1e-12:
            best_val, best_params, wait = val_loss, [p.copy() for p in params], 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_params is not None:
        w1, b1, w2, b2 = best_params

    def predict(x: np.ndarray) -> np.ndarray:
        return np.maximum(x @ w1 + b1, 0.0) @ w2 + b2

    return {
        "r2": r2_explained(z_eval, predict(x_eval)),
        "hidden": hidden,
        "epochs_run": epochs_run,
        "best_val_loss": best_val,
    }


def pool_tokens(evis: np.ndarray, pool: int) -> np.ndarray:
    """Spatial average-pool of frozen E_vis tokens: [rows, V, H, W, D] -> [rows, V, H', W', D]
    flattened to [rows, V*H'*W'*D], with H' = H // pool (requires divisibility).

    Token-level (not pooled-into-C) E_vis is the frozen G3 probe feature (audit line:
    the probe reads the current VLA tokens E_theta(X), never A_hat_0). The C context
    only carries the pooled E_vis, so token-level detail is exactly what conditioning
    leaves available for the probe."""
    evis = np.asarray(evis)
    if evis.ndim != 5:
        raise RuntimeError(f"evis must be [rows, V, H, W, D], got shape {evis.shape}")
    rows, n_views, h, w, d = evis.shape
    if pool <= 1:
        return evis.reshape(rows, -1)
    if h % pool or w % pool:
        raise RuntimeError(f"token grid {h}x{w} not divisible by pool {pool}")
    evis = evis.reshape(rows, n_views, h // pool, pool, w // pool, pool, d)
    return evis.mean(axis=(3, 5)).reshape(rows, -1)


def token_probe_r2(
    probe_x_fit: np.ndarray,
    z_fit: np.ndarray,
    probe_x_eval: np.ndarray,
    z_eval: np.ndarray,
    lam_grid: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0),
    n_sub: int = 2000,
    n_splits: int = 4,
    seed: int = SEED_DEFAULT,
) -> dict:
    """G3 token-level linear probe via DUAL-FORM ridge (kernel ridge on the Gram of a
    seeded row subsample) - required because token features are too wide for the primal
    Gram (d up to V*8*8*1024). Evaluated with row-level K-fold OOF inside the fit pool.

    lambda is picked by the best OOF R2 over the grid (reported per-lambda); with the
    frozen 50% G3 threshold far above the noise floor the mild selection optimism is
    negligible (protocol section 8 G3, oracle L1 two-tier probe)."""
    x_fit = np.asarray(probe_x_fit, dtype=np.float64)
    x_eval = np.asarray(probe_x_eval, dtype=np.float64)
    z_fit = np.asarray(z_fit, dtype=np.float64)
    z_eval = np.asarray(z_eval, dtype=np.float64)
    mean_x, std_x = x_fit.mean(axis=0), x_fit.std(axis=0)
    std_x = np.where(std_x > 1e-12, std_x, 1.0)
    x_fit = (x_fit - mean_x) / std_x
    x_eval = (np.asarray(probe_x_eval, dtype=np.float64) - mean_x) / std_x
    splits = kfold_indices(x_fit.shape[0], n_splits=n_splits, seed=seed)
    per_lam = {}
    for lam in lam_grid:
        oof = np.empty_like(z_fit)
        for i, test_idx in enumerate(splits):
            train_idx = np.concatenate([s for j, s in enumerate(splits) if j != i])
            rng = np.random.default_rng(seed + 11 + i)
            if train_idx.shape[0] > n_sub:
                train_idx = np.sort(rng.choice(train_idx, size=n_sub, replace=False))
            xs, zs = x_fit[train_idx], z_fit[train_idx]
            gram = xs @ xs.T
            alpha = np.linalg.solve(gram + lam * train_idx.shape[0] * np.eye(gram.shape[0]), zs)
            oof[test_idx] = x_fit[test_idx] @ xs.T @ alpha
        per_lam[float(lam)] = r2_explained(z_fit, oof)
    best_lam = max(per_lam, key=lambda lam: per_lam[lam])
    # refit on the full fit pool with the best lambda for the final eval-pool prediction
    rng = np.random.default_rng(seed + 99)
    fit_idx = np.arange(x_fit.shape[0])
    if fit_idx.shape[0] > n_sub:
        fit_idx = np.sort(rng.choice(fit_idx, size=n_sub, replace=False))
    xs, zs = x_fit[fit_idx], z_fit[fit_idx]
    gram = xs @ xs.T
    alpha = np.linalg.solve(gram + best_lam * fit_idx.shape[0] * np.eye(gram.shape[0]), zs)
    eval_r2 = r2_explained(z_eval, x_eval @ xs.T @ alpha)
    return {"r2": eval_r2, "lam": best_lam, "oof_curve": per_lam, "n_sub": int(fit_idx.shape[0])}


# ---------------------------------------------------------------------------
# subspace geometry (protocol section 8 G4) and controls (audit section 3 controls)
# ---------------------------------------------------------------------------


def principal_angles_deg(subspace_a: np.ndarray, subspace_b: np.ndarray) -> np.ndarray:
    """Principal angles (degrees) between two subspaces spanned by the given columns."""
    q_a, _ = np.linalg.qr(np.asarray(subspace_a, dtype=np.float64))
    q_b, _ = np.linalg.qr(np.asarray(subspace_b, dtype=np.float64))
    r = min(q_a.shape[1], q_b.shape[1])
    cosines = np.clip(np.linalg.svd(q_a[:, :r].T @ q_b[:, :r], compute_uv=False), -1.0, 1.0)
    return np.degrees(np.arccos(cosines))


def random_orthogonal_basis(d: int, k: int, seed: int) -> np.ndarray:
    """Random orthonormal k-frame in R^d (Random-subspace control, protocol section 8)."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(d, k)))
    return q


def reversed_time_flow(f_tilde: np.ndarray, episode_ids: np.ndarray) -> np.ndarray:
    """Reversed-time control: the F_tilde row order is reversed WITHIN each episode (the
    valid/kept rows of the episode), so step t's flow is paired with step T-t's action
    while C / A_tilde keep their original row order. This destroys the F_tilde -
    A_tilde temporal alignment and should weaken a causal forward-motion signal; a
    persistent association under reversal indicates the CCA modes are not directional
    (control row in the Stage A report). NOTE: a plain sign flip F_tilde -> -F_tilde is
    NOT a valid reversal - CCA loadings are sign-invariant (SVD u -> +/-u) and the
    linear probe absorbs the sign, so the flip is an identity control; the row order
    must be reversed."""
    f = np.asarray(f_tilde, dtype=np.float64)
    episode_ids = np.asarray(episode_ids)
    out = f.copy()
    for ep in np.unique(episode_ids):
        rows = np.nonzero(episode_ids == ep)[0]
        out[rows] = f[rows[::-1]]
    return out


def nan_anchor_mask(f: np.ndarray) -> np.ndarray:
    """Valid-row mask over a raw flow block: a row is invalid if any cell is non-finite.

    RAFT cache marks episode-tail anchors (step + dt >= length) as NaN
    (cache_raft_flow.py docstring); Stage A excludes them and reports the proportion
    plus the excluded rows' action distribution (audit 2.3 NaN anchor / oracle L2).
    """
    return np.isfinite(np.asarray(f, dtype=np.float64)).all(axis=tuple(range(1, f.ndim)))


# ---------------------------------------------------------------------------
# gate evaluation (protocol section 8 G1-G4) and helpers
# ---------------------------------------------------------------------------


def evaluate_gates(
    delta_r2_star: float,
    delta_r2_star_ci_low: float,
    delta_r2_k_star: int,
    null_max_over_k: np.ndarray,
    probe_r2_best: float,
    median_principal_angle_deg: float,
) -> dict:
    """Frozen gate thresholds - protocol section 8 v1.2 (2026-10-18 user resolution)."""
    null_95 = float(np.percentile(np.asarray(null_max_over_k), 95))
    g1 = bool(delta_r2_star > null_95)
    g2 = bool(delta_r2_star >= DELTA_R2_THRESHOLD and delta_r2_star_ci_low > 0.0)
    g3 = bool(probe_r2_best >= PROBE_THRESHOLD)
    g4 = bool(median_principal_angle_deg < PRINCIPAL_ANGLE_THRESHOLD_DEG)
    return {
        "G1_existence": g1,
        "G2_increment": g2,
        "G3_predictability": g3,
        "G4_fold_consistency": g4,
        "stats": {
            "delta_r2_star": float(delta_r2_star),
            "delta_r2_star_ci_low": float(delta_r2_star_ci_low),
            "delta_r2_k_star": int(delta_r2_k_star),
            "null_95_max_over_k": null_95,
            "probe_r2_best": float(probe_r2_best),
            "median_principal_angle_deg": float(median_principal_angle_deg),
            "thresholds": {
                "delta_r2_min": DELTA_R2_THRESHOLD,
                "probe_r2_min": PROBE_THRESHOLD,
                "median_principal_angle_max_deg": PRINCIPAL_ANGLE_THRESHOLD_DEG,
            },
        },
    }


def mde_standard_error(n_rows: int, k_star: int) -> float:
    """Gate MDE sanity readout (oracle M4): SE(delta R2) approx sqrt(2k)/n_eff for the
    holdout pool. Reported next to G2 so the frozen 1% threshold and the design's power
    can be compared (audit requires the MDE-threshold consistency argument per report)."""
    if n_rows <= 0:
        return float("nan")
    return float(np.sqrt(2.0 * k_star) / n_rows)


def subsample_rows(
    rows_per_episode: dict,
    stride: int,
    seed: int = SEED_DEFAULT,
    offset_seed: bool = True,
) -> np.ndarray:
    """Stride-s subsampling of transition rows within each episode (protocol section 8
    sampling: s=8, sensitivity {4, 16}). Deterministic: the per-episode phase offset is
    seeded; kept rows stay contiguous within episodes so episode-level permutation and
    bootstrap remain valid."""
    rng = np.random.default_rng(seed)
    keep = []
    for ep in sorted(rows_per_episode):
        rows = np.asarray(rows_per_episode[ep])
        offset = int(rng.integers(0, stride)) if offset_seed and rows.size >= stride else 0
        keep.append(rows[offset::stride])
    if not keep:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(keep)).astype(np.int64)
