# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""Synthetic-data unit tests for the RAP-FSD Stage A statistics (protocol v1.2).

Covers the six pre-registered invariants:
1. CCA recovers a known low-rank shared subspace (principal angle near 0).
2. Freedman-Lane permutation test: p < 0.05 under strong synthetic signal,
   p > 0.05 under pure noise (type-I calibration).
3. delta R2(k) is monotone non-decreasing in k.
4. split.py is deterministic; holdout and folds are disjoint with exactly one
   label per episode and nested holdout subsets.
5. Freedman-Lane permutation preserves the C-explainable structure (OOF R2 of
   the C-only prediction is invariant under permutation).
6. NaN anchor rows are removed with an exactly-reported proportion and the
   pipeline does not crash on the remaining rows.
"""

import itertools

import numpy as np
import pytest

from starVLA.analysis.rap_fsd import split, stats


def _shared_subspace_data(n: int, d_f: int, d_a: int, r: int, noise: float, seed: int = 42):
    """Latent z drives both views: F = z Wf + eps, A = z Wa + eps."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, r))
    w_f = rng.normal(size=(r, d_f))
    w_a = rng.normal(size=(r, d_a))
    f = z @ w_f + noise * rng.normal(size=(n, d_f))
    a = z @ w_a + noise * rng.normal(size=(n, d_a))
    return f, a, w_f, w_a


# ---------------------------------------------------------------------------
# 1. CCA subspace recovery
# ---------------------------------------------------------------------------
def test_cca_recovers_known_subspace():
    n, d_f, d_a, r = 1200, 10, 8, 3
    f, a, w_f, _ = _shared_subspace_data(n, d_f, d_a, r, noise=0.05, seed=42)
    cca = stats.fit_cca(f, a, ridge_lambda=0.1)
    assert float(cca.rho[0]) > 0.9, "top canonical correlation should be near 1 for shared signal"
    # Ground-truth shared subspace of standardized F: span{Wf_std} with Wf_std[:, j] = Wf[:, j] / s_j
    std_f = f.std(axis=0)
    w_f_std = w_f.T / std_f[:, None]  # [d_f, r]: standardized shared column space of F
    q_true, _ = np.linalg.qr(w_f_std)
    angles = stats.principal_angles_deg(cca.a_flow[:, :r], q_true)
    assert float(np.median(angles)) < 5.0, f"median subspace angle too large: {angles}"
    assert float(angles.max()) < 12.0, f"max subspace angle too large: {angles}"


# ---------------------------------------------------------------------------
# 2. Permutation test calibration (signal vs pure noise)
# ---------------------------------------------------------------------------
def _episode_c(f_shape_rows: int, n_episodes: int, d_c: int, seed: int):
    rng = np.random.default_rng(seed)
    n = f_shape_rows
    c = rng.normal(size=(n, d_c))
    episode_ids = np.repeat(np.arange(n_episodes), n // n_episodes)
    return c, episode_ids


def _perm_p_value(obs: float, null_max_over_k: np.ndarray) -> float:
    return float((1 + int((null_max_over_k >= obs).sum())) / (null_max_over_k.shape[0] + 1))


@pytest.mark.parametrize("mode", ["signal", "noise"])
def test_permutation_test_calibration(mode):
    n, n_episodes, d_f, d_a, d_c, k = 240, 24, 6, 5, 4, 2
    rng = np.random.default_rng(42 if mode == "signal" else 43)
    c, episode_ids = _episode_c(n, n_episodes, d_c, seed=7)
    z_lat = rng.normal(size=(n, 1))
    w_f = rng.normal(size=(1, d_f))
    w_a = rng.normal(size=(1, d_a))
    w_c = rng.normal(size=(d_c, d_a))
    f = 2.0 * z_lat @ w_f + 0.5 * rng.normal(size=(n, d_f))
    if mode == "signal":
        a = 2.0 * c @ w_c + 3.0 * z_lat @ w_a + 0.5 * rng.normal(size=(n, d_a))
    else:
        a = 2.0 * c @ w_c + 0.5 * rng.normal(size=(n, d_a))
    # Stage A preprocessing: conditionally residualise both blocks on C
    mu_a = stats.fit_conditional_mean(c, a, seed=42)
    mu_f = stats.fit_conditional_mean(c, f, seed=42)
    a_tilde = stats.residualize(a, c, mu_a)
    f_tilde = stats.residualize(f, c, mu_f)
    cca = stats.fit_cca(f_tilde, a_tilde, ridge_lambda=0.1)
    z = cca.project(f_tilde, k)
    obs_curve = stats.delta_r2_oof_curve(z, a_tilde, k_candidates=(k,), n_splits=4, seed=42)
    obs = obs_curve[k]
    null = stats.freedman_lane_null(
        f_tilde,
        a_tilde,
        c,
        episode_ids,
        k_candidates=(k,),
        n_perms=120,
        n_splits=4,
        seed=42,
    )
    p_value = _perm_p_value(obs, null["max_over_k"])
    if mode == "signal":
        assert p_value < 0.05, f"strong signal should reject H0 (obs={obs:.4f}, p={p_value:.4f})"
        assert obs > 0.3
    else:
        assert p_value > 0.05, f"pure noise must not reject H0 (obs={obs:.4f}, p={p_value:.4f})"
        assert obs < 0.10


# ---------------------------------------------------------------------------
# 3. delta R2(k) monotonicity
# ---------------------------------------------------------------------------
def test_delta_r2_monotone():
    f, a, _, _ = _shared_subspace_data(800, 12, 9, 4, noise=0.05, seed=42)
    cca = stats.fit_cca(f, a, ridge_lambda=0.1)
    z = cca.project(f, max(4, 12))
    ks = (1, 2, 3, 4, 6, 8, 12)
    curve = stats.delta_r2_in_sample_curve(z, a, k_candidates=ks)
    values = [curve[k] for k in sorted(curve)]
    for prev, cur in itertools.pairwise(values):
        assert cur >= prev - 1e-10, f"delta R2 curve not monotone: {values}"
    assert values[-1] > 0.5, "full-subspace probe should explain most of the shared variance"


# ---------------------------------------------------------------------------
# 4. split.py determinism and disjointness
# ---------------------------------------------------------------------------
def _synthetic_episodes(n: int, seed: int = 42) -> list[dict]:
    return [
        {
            "episode_id": f"libero_object_demo/ep_{i:05d}",
            "dataset_name": "libero_object_demo",
            "trajectory_id": i,
            "suite": "libero_object",
            "task": f"task_{i % 5}",
        }
        for i in range(n)
    ]


def test_split_deterministic_and_disjoint():
    episodes = _synthetic_episodes(120)
    holdout_sizes = (20, 30, 50)
    labels_a = split.assign_labels(episodes, seed=42, holdout_sizes=holdout_sizes)["episodes"]
    labels_b = split.assign_labels(list(reversed(episodes)), seed=42, holdout_sizes=holdout_sizes)["episodes"]
    assert labels_a == labels_b, "assignment must be a pure function of (seed, episode ids)"

    manifest = split.build_manifest(episodes, seed=42, holdout_sizes=holdout_sizes)
    manifest_2 = split.build_manifest(episodes, seed=42, holdout_sizes=holdout_sizes)
    assert manifest == manifest_2
    lookup_20 = split.label_lookup(manifest, holdout_size=20)
    lookup_30 = split.label_lookup(manifest, holdout_size=30)
    lookup_50 = split.label_lookup(manifest, holdout_size=50)

    for lookup, size in ((lookup_20, 20), (lookup_30, 30), (lookup_50, 50)):
        values = list(lookup.values())
        assert set(values) == {"fold_0", "fold_1", "holdout"}
        assert values.count("holdout") == size, f"holdout size {size} mismatch"
        assert values.count("fold_0") == values.count("fold_1") == (len(episodes) - size) // 2
        holdout_eps = {ep for ep, lab in lookup.items() if lab == "holdout"}
        fold_eps = {ep for ep, lab in lookup.items() if lab != "holdout"}
        assert not (holdout_eps & fold_eps), "holdout must not participate in any fold"
        assert len(holdout_eps | fold_eps) == len(episodes), "each episode exactly one label"

    # nesting: holdout(20) subset of holdout(30) subset of holdout(50)
    h20 = {ep for ep, lab in lookup_20.items() if lab == "holdout"}
    h30 = {ep for ep, lab in lookup_30.items() if lab == "holdout"}
    h50 = {ep for ep, lab in lookup_50.items() if lab == "holdout"}
    assert h20 < h30 < h50

    # fold alternation parity is stable across holdout sizes (sensitivity relabelling)
    for ep, lab20 in lookup_20.items():
        if lab20 != "holdout" and lookup_30[ep] != "holdout" and lookup_50[ep] != "holdout":
            assert lookup_30[ep] == lookup_50[ep] == lab20


# ---------------------------------------------------------------------------
# 5. Freedman-Lane preserves C-explainable structure
# ---------------------------------------------------------------------------
def test_freedman_lane_preserves_c_structure():
    n, n_episodes, d_c = 300, 30, 4
    rng = np.random.default_rng(42)
    c, episode_ids = _episode_c(n, n_episodes, d_c, seed=11)
    w_c = rng.normal(size=(d_c, 6))
    a = 2.0 * c @ w_c + rng.normal(size=(n, 6))
    state = stats.freedman_lane_prepare(a, c, lam=1e-2)

    def oof_c_only_r2(target: np.ndarray) -> float:
        pred = stats.oof_ridge_predict(c, target, lam=1e-2, n_splits=4, seed=42)
        return stats.r2_explained(target, pred)

    before = oof_c_only_r2(a)
    for perm_index in range(3):
        gather = stats.episode_block_permutation(episode_ids, seed=1000 + perm_index)
        a_perm = state.fitted_part + state.residual[gather]
        after = oof_c_only_r2(a_perm)
        assert abs(after - before) < 0.05, (
            f"C-only OOF R2 changed under permutation: before={before:.4f} after={after:.4f}"
        )
        var_ratio = float(state.residual[gather].var() / state.residual.var())
        assert 0.95 < var_ratio < 1.05, f"residual variance not preserved: {var_ratio}"


# ---------------------------------------------------------------------------
# 6. NaN anchor removal with exact reported proportion
# ---------------------------------------------------------------------------
def test_nan_anchor_removal_and_pipeline():
    n, d_f, d_a = 200, 8, 6
    rng = np.random.default_rng(42)
    f = rng.normal(size=(n, d_f))
    nan_rows = rng.choice(n, size=20, replace=False)  # exactly 10% invalid anchors
    f[nan_rows] = np.nan
    mask = stats.nan_anchor_mask(f)
    removal_fraction = float((~mask).sum() / mask.shape[0])
    assert int((~mask).sum()) == 20
    assert abs(removal_fraction - 0.1) < 1e-12

    # downstream pipeline on the kept rows only: must not crash and stay finite
    c = rng.normal(size=(n, 4))
    a = rng.normal(size=(n, d_a))
    mu_a = stats.fit_conditional_mean(c, a, seed=42)
    a_tilde = stats.residualize(a, c, mu_a)
    f_valid = f[mask]
    a_valid = a_tilde[mask]
    cca = stats.fit_cca(f_valid, a_valid, ridge_lambda=0.1)
    z = cca.project(f_valid, 4)
    curve = stats.delta_r2_oof_curve(z, a_valid, k_candidates=(4,), n_splits=4, seed=42)
    assert np.isfinite(list(curve.values())).all()
    excluded_stats = {
        "mean_abs": float(np.abs(a_tilde[~mask]).mean()),
        "mean_norm": float(np.linalg.norm(a_tilde[~mask], axis=1).mean()),
    }
    assert np.isfinite(excluded_stats["mean_abs"]) and np.isfinite(excluded_stats["mean_norm"])


# ---------------------------------------------------------------------------
# 7. G3 token-level E_vis probe: token features decode z where the pooled block
#    (part of C, hence conditioned away) cannot
# ---------------------------------------------------------------------------
def test_token_probe_beats_pooled():
    n, r = 300, 6
    d_ev, v, h, w = 4, 2, 8, 8
    rng = np.random.default_rng(42)
    lat = rng.normal(size=(n, r))
    w_pooled = rng.normal(size=(r, d_ev))  # rank 4 < r: pooled block misses 2 latent dims
    devs = rng.normal(size=(v, h * w, r, d_ev))
    devs = devs - devs.mean(axis=1, keepdims=True)  # per-token deviations, zero mean WITHIN each view
    evis = np.empty((n, v, h, w, d_ev))
    for vi in range(v):
        for hi in range(h):
            for wi in range(w):
                evis[:, vi, hi, wi, :] = lat @ (w_pooled + devs[vi, hi * w + wi])
    pooled = stats.pool_tokens(evis, 8)  # 1x1 per view = pooled over the whole grid
    z = lat @ rng.normal(size=(r, 5)) + 0.05 * rng.normal(size=(n, 5))
    fit_m, ev_m = np.arange(200), np.arange(200, 300)
    r_token = stats.token_probe_r2(evis[fit_m].reshape(200, -1), z[fit_m], evis[ev_m].reshape(100, -1), z[ev_m], seed=42)
    r_pooled = stats.linear_probe_r2(pooled[fit_m], z[fit_m], pooled[ev_m], z[ev_m], seed=42)
    assert r_token["r2"] > 0.8, f"token probe should decode z: {r_token['r2']:.3f}"
    assert r_token["r2"] - r_pooled["r2"] > 0.1, (
        f"token features must beat pooled features: token={r_token['r2']:.3f} pooled={r_pooled['r2']:.3f}"
    )


# ---------------------------------------------------------------------------
# 8. evaluate_gates pure logic: frozen threshold boundaries (protocol v1.2)
# ---------------------------------------------------------------------------
def test_evaluate_gates_thresholds():
    """Gate boundary semantics: G1 strictly >, G2 >= 1% AND ci_low > 0 (strict),
    G3 >= 50% (inclusive), G4 < 45 deg (strict)."""
    null_max = np.linspace(0.0, 1.0, 101)  # 95th percentile = 0.95 (up to float error)
    null_95 = float(np.percentile(null_max, 95))
    assert abs(null_95 - 0.95) < 1e-12

    def gates(delta, ci_low, probe, angle):
        return stats.evaluate_gates(
            delta_r2_star=delta,
            delta_r2_star_ci_low=ci_low,
            delta_r2_k_star=8,
            null_max_over_k=null_max,
            probe_r2_best=probe,
            median_principal_angle_deg=angle,
        )

    base = dict(delta=null_95 + 1e-9, ci_low=0.02, probe=0.5, angle=44.9)
    # G1: strictly above the null 95th percentile (equality fails)
    assert gates(**base)["G1_existence"]
    assert not gates(**{**base, "delta": null_95})["G1_existence"]

    # G2: >= 1% AND bootstrap CI low strictly > 0 (neither condition alone suffices)
    assert gates(delta=0.01, ci_low=1e-9, probe=0.0, angle=90.0)["G2_increment"]
    assert not gates(delta=0.0099, ci_low=0.5, probe=0.0, angle=90.0)["G2_increment"]
    assert not gates(delta=0.01, ci_low=0.0, probe=0.0, angle=90.0)["G2_increment"]
    assert not gates(delta=0.02, ci_low=-1e-9, probe=0.0, angle=90.0)["G2_increment"]

    # G3: >= 50% explained variance (equality passes)
    assert gates(delta=0.02, ci_low=0.01, probe=0.5, angle=90.0)["G3_predictability"]
    assert not gates(delta=0.02, ci_low=0.01, probe=0.4999, angle=90.0)["G3_predictability"]

    # G4: median principal angle strictly below 45 deg (equality fails)
    assert gates(delta=0.02, ci_low=0.01, probe=0.0, angle=45.0 - 1e-9)["G4_fold_consistency"]
    assert not gates(delta=0.02, ci_low=0.01, probe=0.0, angle=45.0)["G4_fold_consistency"]

    # all-pass and all-fail combinations
    all_pass = gates(null_95 + 1e-9, 0.02, 0.5, 44.9)
    assert all(all_pass[k] for k in ("G1_existence", "G2_increment", "G3_predictability", "G4_fold_consistency"))
    all_fail = gates(null_95, 0.0, 0.4999, 45.0)
    assert not any(all_fail[k] for k in ("G1_existence", "G2_increment", "G3_predictability", "G4_fold_consistency"))


# ---------------------------------------------------------------------------
# 7. Controls accept folds of unequal length (regression: cluster crash at
#    np.mean over per-fold pool residuals of shape (n_foldA,) vs (n_foldB,))
# ---------------------------------------------------------------------------
def test_controls_unequal_fold_lengths():
    from types import SimpleNamespace

    from starVLA.analysis.rap_fsd import run_stage_a as rsa

    rng = np.random.default_rng(0)
    d_f, d_c, d_a, k_star = 6, 4, 2, 2
    args = SimpleNamespace(seed=0, cca_lambda=0.1, probe_lam=0.1)

    def make_pool(label: str, n: int, n_ep: int):
        c = rng.normal(size=(n, d_c))
        f = rng.normal(size=(n, d_f))
        r_a = rng.normal(size=(n, d_a))
        ep = rng.integers(0, n_ep, size=n)
        return rsa.RowPool(label=label, f=f, r_a=r_a, c=c, episode_id=ep)

    pools = {
        lab: make_pool(lab, n, 3) for lab, n in zip(rsa.FOLD_LABELS, (40, 33), strict=True)
    }
    holdout = make_pool("holdout", 25, 3)
    mu_a_pool = stats.fit_conditional_mean(pools["fold_0"].c, pools["fold_0"].r_a, seed=0)

    fold_models = {}
    for i, (lab, pool) in enumerate(pools.items()):
        mu_f = stats.fit_conditional_mean(pool.c, pool.f, seed=i)
        f_tilde = stats.residualize(pool.f, pool.c, mu_f)
        a_tilde = stats.residualize(pool.r_a, pool.c, mu_a_pool)
        q, _ = np.linalg.qr(rng.normal(size=(d_f, k_star + 1)))
        fold_models[lab] = SimpleNamespace(
            mu_f=mu_f,
            a_tilde=a_tilde,
            f_tilde=f_tilde,
            cca=stats.fit_cca(f_tilde, a_tilde, ridge_lambda=args.cca_lambda),
            rrr={
                "u": q,
                "s": np.abs(rng.normal(size=k_star + 1)),
                "vt": np.linalg.qr(rng.normal(size=(k_star + 1, d_a)))[0],
            },
        )

    mu_a_h = stats.fit_conditional_mean(holdout.c, holdout.r_a, seed=0)
    a_tilde_avg = holdout.r_a - mu_a_h.predict(holdout.c)

    result = rsa._controls(fold_models, pools, holdout, k_star, args, a_tilde_avg)
    for arm in ("rrr", "random_subspace", "reversed_time"):
        assert np.isfinite(result[arm]["delta_r2_k_star"]), arm
