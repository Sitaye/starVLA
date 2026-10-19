# Copyright 2026 StarVLA Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OE-VLA gate-A analysis: pre-registered gates G0, A1, A2, A4, A6.

Reads branches/*.jsonl + states.json (branch.py outputs), fits per-state
Jacobians from probe rollouts, evaluates the outcome-metric hypothesis against
the trace-only isotropic baseline and the contact-free analytic baseline, and
applies the gate thresholds from the protocol (section 5). Pure numpy; no
scipy/sklearn. All directional statistics live on the post-horizon continuous
block; success is the tail metric.
"""

import json
import math
from pathlib import Path

import numpy as np

from starVLA.analysis.oevla import descriptor, estimate, lerobot_source

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 4242
NOISE_FLOOR_FACTOR = 0.1
LINEAR_RATIO_RANGE = (2.5, 6.0)
ACTION_DIM = 7

THRESHOLDS = {
    "A1_median_rho": 0.40,
    "A1_ci_low": 0.15,
    "A2_delta_rho": 0.10,
    "A4_delta_rho": 0.10,
    "A6_delta_rho": 0.25,
    "G0_valid_rate": 0.95,
}


def rankdata(x):
    """Average-tie ranks (scipy-free)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc, yc = x - np.mean(x), y - np.mean(y)
    den = math.sqrt(float(np.dot(xc, xc)) * float(np.dot(yc, yc)))
    return float(np.dot(xc, yc) / den) if den > 0 else 0.0


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def cluster_bootstrap_ci(fn, clusters, n_reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    """Percentile CI resampling whole clusters (tasks) with replacement."""
    clusters = list(clusters)
    stats = np.empty(n_reps)
    rng = np.random.default_rng(seed)
    for r in range(n_reps):
        idx = rng.integers(0, len(clusters), len(clusters))
        stats[r] = fn([clusters[i] for i in idx])
    finite = stats[np.isfinite(stats)]
    if len(finite) < 0.5 * n_reps:
        return float("nan"), float("nan"), len(stats) - len(finite)
    return (
        float(np.percentile(finite, 2.5)),
        float(np.percentile(finite, 97.5)),
        len(stats) - len(finite),
    )


def load_states(out_dir):
    """states.json entries keyed by state id."""
    try:
        payload = json.loads((Path(out_dir) / "states.json").read_text())["states"]
    except (OSError, KeyError, ValueError) as e:
        raise ValueError(f"{out_dir}: missing or malformed states.json ({e})") from e
    for e in payload:
        e["task_id"] = f"{e['suite']}__{e['task']}"
        e["state_id"] = _state_id(e)
    return payload


def load_branches(out_dir):
    branches = {}
    for p in sorted((Path(out_dir) / "branches").glob("*.jsonl")):
        try:
            records = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        except ValueError as e:
            raise ValueError(f"{p}: malformed jsonl ({e})") from e
        branches[p.stem] = records
    return branches


def fit_state_jacobian(records, weights, horizon="phi_post"):
    """Fit J^T from the eps=1 probe rollouts of one state.

    Baselines give the noise-floor spread and reference phi; eps=1 records
    (train + val folds) are fit and scored; eps in {0.5, 2} are returned as
    linear-regime rows. Returns None when a state is unusable.
    """
    base = [r for r in records if r["kind"] == "baseline" and r["valid"]]
    pert = [r for r in records if r["kind"] == "perturb" and r["valid"]]
    if len(base) < 2 or len(pert) < 10:
        return None
    phi0 = np.asarray(base[0][horizon], dtype=float)
    spread = 0.0
    for b in base[1:]:
        d0 = np.asarray(b[horizon], dtype=float) - phi0
        spread = max(spread, float(d0 @ (weights * d0)))

    main = [r for r in pert if r["eps"] == 1.0 and r["fold"] in ("train", "val")]
    E = np.array([r["dir"] for r in main])
    D = np.array([np.asarray(r[horizon]) - phi0 for r in main])
    J = estimate.fit_jacobian(E, D)
    G = estimate.outcome_metric(J, weights)

    G_an, J_an = None, None
    snap = next((r["snapshot"] for r in records if r.get("snapshot")), None)
    if snap is not None and "chunk" in snap:
        layout = {"object_names": sorted(snap["objects"].keys())}
        chunk = np.asarray(snap["chunk"], dtype=float).reshape(-1, ACTION_DIM)
        J_an = estimate.analytic_jacobian(_hydrate_snapshot(snap), chunk, layout)
        G_an = estimate.outcome_metric(J_an, weights)

    rows = [
        {
            "eps": rec["eps"],
            "fold": rec["fold"],
            "e": e,
            "measured": float(d @ (weights * d)),
            "cf": estimate.direction_cost(G, e),
            "trace": estimate.trace_only_cost(G, e),
            "an": estimate.direction_cost(G_an, e) if G_an is not None else None,
            "success": rec["success"],
        }
        for rec, e, d in zip(main, E, D, strict=True)
    ]
    lin_rows = [
        {
            "eps": r["eps"],
            "measured": float(
                (np.asarray(r[horizon]) - phi0) @ (weights * (np.asarray(r[horizon]) - phi0))
            ),
        }
        for r in pert if r["fold"] == "lin"
    ]
    return {
        "G": G,
        "G_an": G_an,
        "rows": rows,
        "lin_rows": lin_rows,
        "baseline_norm": descriptor.normalized_features(phi0, _layout_from_dim(len(phi0)))[:6],
        "baseline_spread": spread,
        "baseline_success": [b["success"] for b in base],
        "perturb_success": [r["success"] for r in pert],
        "n_valid": len(pert),
        "n_invalid": len(records) - len(base) - len(pert),
    }


def _layout_from_dim(m):
    n_obj = (m - 7) // 9
    return {"object_names": [f"obj{i}" for i in range(n_obj)]}


def _hydrate_snapshot(snap):
    return {
        "eef_pos": np.asarray(snap["eef_pos"], dtype=float),
        "eef_quat": np.asarray(snap["eef_quat"], dtype=float),
        "gripper_width": snap["gripper_width"],
        "grasp_name": snap["grasp_name"],
        "objects": {
            k: (np.asarray(v["pos"], dtype=float), np.asarray(v["quat"], dtype=float))
            for k, v in snap["objects"].items()
        },
    }


def rho(rows, model):
    """Spearman correlation between measured directional costs and a model."""
    pairs = [(r["measured"], r[model]) for r in rows if r[model] is not None]
    if len(pairs) < 3:
        return float("nan")
    xs, ys = (np.array(p) for p in zip(*pairs, strict=True))
    return spearman(xs, ys)


def analyze(out_dir, report=True):
    entries = load_states(out_dir)
    branches = load_branches(out_dir)
    per_state, per_task = {}, {}
    for e in entries:
        recs = branches.get(e["state_id"])
        if recs is None:
            continue
        phi0 = next((r for r in recs if r["kind"] == "baseline"), None)
        if phi0 is None:
            continue
        weights = descriptor.metric_weights(_layout_from_dim(len(phi0["phi_post"])))
        fit = fit_state_jacobian(recs, weights)
        if fit is not None:
            per_state[e["state_id"]] = fit
            per_task.setdefault(e["task_id"], []).append(e["state_id"])
    if not per_state:
        raise ValueError(f"{out_dir}: no usable states fitted")

    results = {"n_states": len(entries), "n_fit": len(per_state), "gates": {}}
    results["gates"]["G0"] = _gate_g0(per_state, len(entries))
    results["gates"]["A1"] = _gate_a1(per_state, per_task)
    results["gates"]["A2"] = _gate_a2(per_state, per_task)
    results["gates"]["A4"] = _gate_a4(per_state, per_task)
    results["gates"]["A6"] = _gate_a6(per_state, entries, per_task)
    results["linearity"] = _linearity(per_state)
    results["breakdowns"] = {
        "per_suite": _suite_breakdown(entries, per_state),
        "per_third": _third_breakdown(entries, per_state),
        "effective_rank": _median_rank_breakdown(per_state),
        "tail": _tail_breakdown(per_state),
    }
    gate_keys = ["G0", "A1", "A2", "A4", "A6"]
    results["verdict"] = {
        "gate_b_go": bool(all(results["gates"][k]["pass"] for k in gate_keys)),
        "failed": [k for k in gate_keys if not results["gates"][k]["pass"]],
    }

    (Path(out_dir) / "results.json").write_text(
        json.dumps(results, indent=2, default=_json_default)
    )
    if report:
        (Path(out_dir) / "report.md").write_text(render_report(results))
    return results


def _median_stat(fn, per_state, per_task, cluster_sample):
    vals = [fn(per_state[s]) for t in cluster_sample for s in per_task.get(t, []) if s in per_state]
    return float(np.nanmedian(vals)) if vals else float("nan")


def _mean_delta(x, y):
    pairs = [(a, b) for a, b in zip(x, y, strict=True) if not (math.isnan(a) or math.isnan(b))]
    return float(np.mean([a - b for a, b in pairs])) if pairs else float("nan")


def _gate_g0(per_state, n_selected):
    spreads = [f["baseline_spread"] for f in per_state.values()]
    costs = [r["measured"] for f in per_state.values() for r in f["rows"]]
    median_cost = float(np.median(costs)) if costs else 0.0
    p95 = float(np.percentile(spreads, 95)) if spreads else float("nan")
    n_tot = sum(f["n_valid"] + f["n_invalid"] for f in per_state.values())
    valid_rate = sum(f["n_valid"] for f in per_state.values()) / max(1, n_tot)
    recovery = len(per_state) / max(1, n_selected)
    g0 = {
        "median_perturb_cost": median_cost,
        "p95_baseline_spread": p95,
        "valid_rate": valid_rate,
        "state_recovery_rate": recovery,
        "noise_ok": bool(spreads and p95 < NOISE_FLOOR_FACTOR * median_cost),
        "valid_rate_ok": bool(valid_rate >= THRESHOLDS["G0_valid_rate"]),
        "recovery_ok": bool(recovery >= THRESHOLDS["G0_valid_rate"]),
    }
    g0["pass"] = bool(g0["noise_ok"] and g0["valid_rate_ok"] and g0["recovery_ok"])
    return g0


def _gate_a1(per_state, per_task):
    fn = lambda f: rho(f["rows"], "cf")  # noqa: E731
    median = _median_stat(fn, per_state, per_task, list(per_task.keys()))
    lo, hi, n_nan = cluster_bootstrap_ci(
        lambda cl: _median_stat(fn, per_state, per_task, cl), list(per_task.keys())
    )
    return {
        "median_rho": median,
        "ci": [lo, hi],
        "n_nan_reps": n_nan,
        "pass": bool(
            median >= THRESHOLDS["A1_median_rho"] and lo > THRESHOLDS["A1_ci_low"]
        ),
    }


def _gate_a2(per_state, per_task):
    cf, tr = _fold_rhos(per_state, "val")

    def _stat(cluster_sample):
        ids = {s for t in cluster_sample for s in per_task.get(t, [])}
        sel = [s for s in ids if s in cf]
        return _mean_delta([cf[s] for s in sel], [tr[s] for s in sel])

    lo, hi, n_nan = cluster_bootstrap_ci(_stat, list(per_task.keys()))
    delta = _mean_delta(list(cf.values()), list(tr.values()))
    return {
        "delta_rho": delta,
        "ci": [lo, hi],
        "n_nan_reps": n_nan,
        "pass": bool(
            delta >= THRESHOLDS["A2_delta_rho"] and lo > 0 and hi > THRESHOLDS["A2_delta_rho"]
        ),
    }


def _gate_a4(per_state, per_task):
    cf = {s: rho(f["rows"], "cf") for s, f in per_state.items() if f["G_an"] is not None}
    an = {s: rho(f["rows"], "an") for s, f in per_state.items() if f["G_an"] is not None}

    def _stat(cluster_sample):
        ids = {s for t in cluster_sample for s in per_task.get(t, [])}
        sel = [s for s in ids if s in cf]
        return _mean_delta([cf[s] for s in sel], [an[s] for s in sel])

    lo, hi, n_nan = cluster_bootstrap_ci(_stat, list(per_task.keys()))
    delta = _mean_delta(list(cf.values()), list(an.values()))
    return {
        "delta_rho": delta,
        "ci": [lo, hi],
        "n_nan_reps": n_nan,
        "pass": bool(
            delta >= THRESHOLDS["A4_delta_rho"] and lo > 0 and hi > THRESHOLDS["A4_delta_rho"]
        ),
    }


def _gate_a6(per_state, entries, per_task):
    transfer = [e["state_id"] for e in entries if e["transfer"] and e["state_id"] in per_state]
    if len(transfer) < 2:
        return {"delta_rho": float("nan"), "n_transfer": len(transfer), "ci": [float("nan")] * 2, "pass": False}
    cf, tr = _nn_transfer(per_state, entries, transfer)

    def _stat(cluster_sample):
        ids = {s for t in cluster_sample for s in per_task.get(t, [])}
        sel = [s for s in transfer if s in ids]
        if len(sel) < 2:
            return float("nan")
        c, t_ = _nn_transfer(per_state, entries, sel)
        return _mean_delta(c, t_)

    lo, hi, n_nan = cluster_bootstrap_ci(_stat, list(per_task.keys()))
    delta = _mean_delta(cf, tr)
    return {
        "delta_rho": delta,
        "n_transfer": len(transfer),
        "ci": [lo, hi],
        "n_nan_reps": n_nan,
        "pass": bool(
            delta >= THRESHOLDS["A6_delta_rho"] and lo > 0 and hi > THRESHOLDS["A6_delta_rho"]
        ),
    }


def _fold_rhos(per_state, fold):
    cf, tr = {}, {}
    for s, f in per_state.items():
        rows = [r for r in f["rows"] if r["fold"] == fold]
        if len(rows) >= 3:
            cf[s] = rho(rows, "cf")
            tr[s] = rho(rows, "trace")
    return cf, tr


def _nn_transfer(per_state, entries, transfer_ids):
    """Nearest-neighbor Jacobian transfer: each transfer state borrows G from
    its nearest fit-state (same suite, normalized eef-pose distance, A6)."""
    feats = {s: f["baseline_norm"] for s, f in per_state.items()}
    transfer_set = set(transfer_ids)
    cf, tr = [], []
    for s in transfer_ids:
        suite = s.split("__")[0]
        fit_ids = [
            t
            for t in per_state
            if t not in transfer_set and t.split("__")[0] == suite
        ]
        if not fit_ids:
            continue
        nn = min(fit_ids, key=lambda t: float(np.linalg.norm(feats[s] - feats[t])))
        rows = per_state[s]["rows"]
        G_nn = per_state[nn]["G"]
        cf.append(
            spearman(
                [r["measured"] for r in rows],
                [estimate.direction_cost(G_nn, r["e"]) for r in rows],
            )
        )
        tr.append(
            spearman(
                [r["measured"] for r in rows],
                [estimate.trace_only_cost(G_nn, r["e"]) for r in rows],
            )
        )
    return cf, tr


def _linearity(per_state):
    """Cost scaling c(2 eps)/c(eps) per state; ~4 in the local regime."""
    ratios = []
    for f in per_state.values():
        by_eps = {}
        for r in f["lin_rows"]:
            by_eps.setdefault(round(r["eps"], 6), []).append(r["measured"])
        lo, hi = by_eps.get(1.0), by_eps.get(2.0)
        if lo and hi and np.mean(lo) > 0:
            ratios.append(float(np.mean(hi) / np.mean(lo)))
    med = float(np.nanmedian(ratios)) if ratios else float("nan")
    return {
        "median_c2_over_c1": med,
        "range": list(LINEAR_RATIO_RANGE),
        "in_range": bool(
            ratios and LINEAR_RATIO_RANGE[0] <= med <= LINEAR_RATIO_RANGE[1]
        ),
    }


def _median_rank_breakdown(per_state):
    cf = [estimate.effective_rank(f["G"]) for f in per_state.values()]
    an = [estimate.effective_rank(f["G_an"]) for f in per_state.values() if f["G_an"] is not None]
    return {
        "median": float(np.median(cf)) if cf else None,
        "median_analytic": float(np.median(an)) if an else None,
    }


def _suite_breakdown(entries, per_state):
    out = {}
    for suite in sorted({e["suite"] for e in entries}):
        ids = [
            e["state_id"]
            for e in entries
            if e["suite"] == suite and e["state_id"] in per_state
        ]
        if ids:
            out[suite] = {
                "n": len(ids),
                "median_rho": float(
                    np.nanmedian([rho(per_state[s]["rows"], "cf") for s in ids])
                ),
                "success_rate": float(
                    np.mean([np.mean(per_state[s]["perturb_success"]) for s in ids])
                ),
            }
    return out


def _third_breakdown(entries, per_state):
    third_of = {e["state_id"]: int(e.get("third", 0)) for e in entries}
    groups = {0: [], 1: [], 2: []}
    for s, f in per_state.items():
        groups[third_of.get(s, 0)].append(rho(f["rows"], "cf"))
    out = {}
    for k, v in groups.items():
        name = ("early", "mid", "late")[k]
        out[name] = {
            "n": len(v),
            "median_rho": float(np.nanmedian(v)) if v else None,
        }
    return out


def _tail_breakdown(per_state):
    """Tail metric: baseline success rate and perturbation-induced success flips."""
    base_rate = [float(np.mean(f["baseline_success"])) for f in per_state.values()]
    flips = []
    for f in per_state.values():
        base_on = np.mean(f["baseline_success"]) > 0.5
        flips.append(float(np.mean([r["success"] != base_on for r in f["rows"]])))
    return {
        "baseline_success_rate": float(np.mean(base_rate)),
        "flip_rate": float(np.mean(flips)),
    }


def _state_id(e):
    return f"{e['suite']}__{lerobot_source.norm_text(e['task'])}__ep{e['episode_index']}__t{e['t0']}"


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def render_report(r):
    g = r["gates"]
    failed = ", ".join(r["verdict"]["failed"]) or "none"
    lines = [
        "# OE-VLA gate A report",
        "",
        f"- states: {r['n_states']} (fit {r['n_fit']})",
        f"- verdict: **{'GO to gate B' if r['verdict']['gate_b_go'] else 'NO-GO'}** (failed: {failed})",
        "",
        "| gate | statistic | threshold | pass |",
        "|---|---|---|---|",
    ]
    lines.append(
        _gate_row(
            "G0",
            f"valid_rate={g['G0']['valid_rate']:.3f}, recovery={g['G0']['state_recovery_rate']:.3f},"
            f" noise_ok={g['G0']['noise_ok']}",
            ">=0.95 / >=0.95 / spread<0.1·cost",
            g["G0"]["pass"],
        )
    )
    lines.append(
        _gate_row(
            "A1",
            f"median rho={g['A1']['median_rho']:.3f}, CI low={g['A1']['ci'][0]:.3f}",
            ">=0.40 / >0.15",
            g["A1"]["pass"],
        )
    )
    lines.append(
        _gate_row(
            "A2",
            f"delta rho (val)={g['A2']['delta_rho']:.3f}, CI low={g['A2']['ci'][0]:.3f}",
            ">=0.10 / >0",
            g["A2"]["pass"],
        )
    )
    lines.append(
        _gate_row(
            "A4",
            f"delta rho (vs analytic)={g['A4']['delta_rho']:.3f}, CI low={g['A4']['ci'][0]:.3f}",
            ">=0.10 / >0",
            g["A4"]["pass"],
        )
    )
    lines.append(
        _gate_row(
            "A6",
            f"delta rho (NN transfer)={g['A6']['delta_rho']:.3f}, CI low={g['A6']['ci'][0]:.3f}",
            ">=0.25 / >0",
            g["A6"]["pass"],
        )
    )
    lines += [
        "",
        f"- linearity: median c(2e)/c(e) = {r['linearity']['median_c2_over_c1']:.2f}"
        f" (target {r['linearity']['range']})",
        f"- effective rank (median): CF {r['breakdowns']['effective_rank']['median']},"
        f" analytic {r['breakdowns']['effective_rank']['median_analytic']}",
        "",
        "## per-suite",
        "",
    ]
    for suite, b in r["breakdowns"]["per_suite"].items():
        lines.append(
            f"- {suite}: n={b['n']}, median rho={b['median_rho']:.3f}, success={b['success_rate']:.3f}"
        )
    return "\n".join(lines) + "\n"


def _gate_row(name, stat, thr, ok):
    return f"| {name} | {stat} | {thr} | {ok} |"
