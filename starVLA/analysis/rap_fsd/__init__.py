# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""RAP-FSD Stage A offline validation package.

Implements the pre-registered Stage A protocol v1.2
(docs/research/ac_audit_rapfsd_math_20261018.md section 8):

- :mod:`.split`       component 1: deterministic seed=42 episode split
                      (K=2 folds + holdout 2500, sensitivity 2000/3000).
- :mod:`.export_pi0`  component 2: pi0 prediction / context / optional E_vis export.
- :mod:`.stats`       component 3 core: ridge conditional means, hand-written CCA,
                      delta-R2 curves, Freedman-Lane permutation null, block
                      bootstrap, predictability probes, gates (pure numpy/scipy).
- :mod:`.run_stage_a` component 3 CLI: manifest + shards + flow cache -> results.json
                      + report.md with the go/no-go decision.
- :mod:`.materialize` component 4: fold-OOF z.npy targets, top-k mode visualizations.

All statistics are CPU-only numpy/scipy (sklearn is not a repo dependency; ridge and
CCA are implemented here). Heavy torch imports are lazy so this package is importable
on machines without GPU / data.
"""

from .split import assign_labels, build_manifest, episode_id_of, label_lookup, load_manifest
from .stats import (
    CCAProjection,
    ConditionalMean,
    RidgeRegressor,
    block_bootstrap_delta_r2,
    delta_r2_oof_curve,
    evaluate_gates,
    fit_cca,
    fit_conditional_mean,
    fit_linear_probe,
    fit_rrr,
    freedman_lane_null,
    linear_probe_r2,
    mlp_probe_r2,
    pool_tokens,
    principal_angles_deg,
    r2_explained,
    residualize,
    token_probe_r2,
)

__all__ = [
    "CCAProjection",
    "ConditionalMean",
    "RidgeRegressor",
    "assign_labels",
    "block_bootstrap_delta_r2",
    "build_manifest",
    "delta_r2_oof_curve",
    "episode_id_of",
    "evaluate_gates",
    "fit_cca",
    "fit_conditional_mean",
    "fit_linear_probe",
    "fit_rrr",
    "freedman_lane_null",
    "label_lookup",
    "linear_probe_r2",
    "load_manifest",
    "mlp_probe_r2",
    "pool_tokens",
    "principal_angles_deg",
    "r2_explained",
    "residualize",
    "token_probe_r2",
]
