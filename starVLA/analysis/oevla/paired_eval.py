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
"""Paired comparison of two eval trees: the gate's decision statistic.

The gate (design doc 6.11) is a difference against the *reference arms*, not against the
baseline, and the pre-registered statistic is a task-level paired bootstrap CI. Nothing in
the repo computed it: `parallel_eval/aggregate_results.py` only sums shard counts, and a
shard entry is a `{total_count, success_count}` pair keyed by task id (LIBERO) or by
disturbance category (LIBERO-Plus) -- the per-episode records are gone by then. So the
finest stratum that survives to disk is that key, and `(suite, key)` is the resampling unit.

Three consequences, written down so the numbers cannot be read as something stronger:

- The bootstrap resamples keys, not episodes. It buys a design-effect-aware CI for the
  pooled success-rate difference; it does not recover per-episode pairing. With ~40 LIBERO
  keys or ~30 LIBERO-Plus keys the percentile CI is coarse-grained but honest.
- The p-value is the count-only analogue of McNemar. Under H0, both arms' episodes inside a
  stratum are exchangeable, so the null draw of arm A's successes is hypergeometric over the
  stratum's pooled outcomes. Episode-level pairing would be strictly more powerful and is
  not available from these artifacts, so p is conservative.
- `--min-delta` and every `*_pp` field are percentage points, the units the gate is written
  in; success rates stay fractions.

So this must be frozen before the numbers exist (design doc 6.14): choosing the estimator
after seeing the results is exactly the freedom the gate was written to remove.
"""

import argparse
import fnmatch
import json
import sys
from pathlib import Path

import numpy as np

PER_SUITE_FILE = "task_results.json"


def load_arm(root, suites=None, shard_glob="*.json", skip_globs=(), expected_per_suite=None):
    """Return `{(suite, key): (episodes, successes)}` summed over the selected shards of a suite.

    Shards are disjoint task ranges, so keys accumulate rather than collide; a key that
    appears twice is summed, which is what a re-run of an overlapping shard would also be.

    `logs/<suite>/` is shared by every endpoint that ever scored this checkpoint, and the two
    endpoints do not agree on what a key means: LIBERO writes `<start>_<end>.json` keyed by task
    id, LIBERO-Plus writes `<start>_to_<end>.json` keyed by disturbance category. Globbing all of
    it mixes both, doubles the episode count and breaks the key alignment, so `shard_glob` /
    `skip_globs` select one endpoint and `expected_per_suite` refuses an implausible total.
    """
    logs = Path(root) / "logs"
    if not logs.is_dir():
        raise SystemExit(f"{root}: no logs/ directory (expected <ckpt>/logs/<suite>/*.json)")
    present = sorted(p.name for p in logs.iterdir() if p.is_dir())
    if suites is None:
        suites = present
    missing = [s for s in suites if s not in present]
    if missing:
        raise SystemExit(f"{root}: missing suite(s) {missing}; present {present}")
    strata = {}
    for suite in suites:
        files = sorted(
            p
            for p in (logs / suite).glob(shard_glob)
            if p.name != PER_SUITE_FILE
            and not p.name.startswith(".tmp_")
            and not any(fnmatch.fnmatch(p.name, g) for g in skip_globs)
        )
        if not files:
            raise SystemExit(f"{logs / suite}: no shard JSONs match {shard_glob!r} (skipping {list(skip_globs)})")
        for path in files:
            try:
                shard = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise SystemExit(f"cannot read shard {path}: {exc}") from exc
            if not isinstance(shard, dict) or all(not isinstance(v, dict) for v in shard.values()):
                # A per-suite summary (`overall_sr.json` and friends) reuses the count names at the
                # top level; it aggregates the shards, so counting it would double-count the suite.
                print(f"[paired] ignoring non-shard {path}", file=sys.stderr)
                continue
            for key, entry in shard.items():
                try:
                    n, s = int(entry["total_count"]), int(entry["success_count"])
                except (TypeError, KeyError, ValueError) as exc:
                    raise SystemExit(f"{path}: malformed entry {key!r}: {exc}") from exc
                if not 0 <= s <= n:
                    raise SystemExit(f"{path}: {key!r} has successes={s} outside [0, {n}]")
                prev_n, prev_s = strata.get((suite, key), (0, 0))
                strata[(suite, key)] = (prev_n + n, prev_s + s)
    per_suite = {s: sum(n for (suite, _), (n, _) in strata.items() if suite == s) for s in suites}
    print(f"[paired] {Path(root).name}: " + " ".join(f"{s}={per_suite[s]}" for s in suites), file=sys.stderr)
    if expected_per_suite is not None:
        wrong = {s: n for s, n in per_suite.items() if n != expected_per_suite}
        if wrong:
            raise SystemExit(
                f"{root}: suite episode counts {wrong} differ from the expected {expected_per_suite} "
                f"(select one endpoint with --shard-glob/--skip-glob)"
            )
    return strata


def list_shards(root, suites=None):
    """Print every JSON file in every suite dir with its episode total, to expose a mixed logs/ tree."""
    logs = Path(root) / "logs"
    present = sorted(p.name for p in logs.iterdir() if p.is_dir())
    print(f"\n[root] {root}")
    print(f"{'suite':<16} {'file':<26} {'episodes':>8} {'keys':>5}")
    for suite in suites or present:
        for path in sorted((logs / suite).glob("*.json")):
            try:
                shard = json.loads(path.read_text())
                values = list(shard.values()) if isinstance(shard, dict) else []
                total = sum(int(v["total_count"]) for v in values if isinstance(v, dict) and "total_count" in v)
            except (OSError, json.JSONDecodeError, TypeError, KeyError, ValueError):
                total = -1
            print(f"{suite:<16} {path.name:<26} {total:>8} {len(values):>5}")


def align(strata_a, strata_b, name_a="arm A", name_b="arm B"):
    """Sort keys and refuse to proceed unless both arms scored the same episodes."""
    keys = sorted(set(strata_a) | set(strata_b))
    absent = [k for k in keys if k not in strata_a or k not in strata_b]
    if absent:
        raise SystemExit(f"{name_a} and {name_b} do not share {len(absent)} key(s), e.g. {absent[:3]}")
    unequal = [k for k in keys if strata_a[k][0] != strata_b[k][0]]
    if unequal:
        raise SystemExit(
            f"{len(unequal)} key(s) have different episode counts between {name_a} and {name_b}, "
            f"e.g. {[(k, strata_a[k][0], strata_b[k][0]) for k in unequal[:3]]}"
        )
    n = np.array([strata_a[k][0] for k in keys], dtype=float)
    sa = np.array([strata_a[k][1] for k in keys], dtype=float)
    sb = np.array([strata_b[k][1] for k in keys], dtype=float)
    return keys, n, sa, sb


def pooled_delta(n, sa, sb):
    """Pooled success-rate difference in percentage points (trial-weighted)."""
    return float((sa.sum() - sb.sum()) / n.sum() * 100.0)


def paired_bootstrap(n, sa, sb, draws, seed):
    """Percentile CI of the pooled difference, resampling `(suite, key)` strata."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n.size, size=(draws, n.size))
    wn, wa, wb = n[idx].sum(1), sa[idx].sum(1), sb[idx].sum(1)
    keep = wn > 0
    diffs = (wa[keep] - wb[keep]) / wn[keep] * 100.0
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi), float(diffs.std(ddof=1))


def stratified_permutation(n, sa, sb, draws, seed):
    """Monte-Carlo p for equal success probability within every stratum (McNemar analogue)."""
    rng = np.random.default_rng(seed)
    counts, pooled = n.astype(np.int64), (sa + sb).astype(np.int64)
    null_a = rng.hypergeometric(pooled, 2 * counts - pooled, counts, size=(draws, n.size))
    total = n.sum()
    null_delta = (null_a.sum(1) - (pooled.sum() - null_a.sum(1))) / total * 100.0
    observed = pooled_delta(n, sa, sb)
    extreme = int((np.abs(null_delta) >= abs(observed) - 1e-9).sum())
    return (extreme + 1) / (draws + 1)


def compare(strata_a, strata_b, name_a, name_b, draws=10000, seed=42):
    keys, n, sa, sb = align(strata_a, strata_b, name_a, name_b)
    delta = pooled_delta(n, sa, sb)
    lo, hi, se = paired_bootstrap(n, sa, sb, draws, seed)
    per_suite = {}
    for suite in sorted({k[0] for k in keys}):
        rows = [i for i, k in enumerate(keys) if k[0] == suite]
        per_suite[suite] = {
            "episodes": int(n[rows].sum()),
            "delta_pp": pooled_delta(n[rows], sa[rows], sb[rows]),
            "sr_a": float(sa[rows].sum() / n[rows].sum()),
            "sr_b": float(sb[rows].sum() / n[rows].sum()),
        }
    return {
        "a": name_a,
        "b": name_b,
        "episodes": int(n.sum()),
        "sr_a": float(sa.sum() / n.sum()),
        "sr_b": float(sb.sum() / n.sum()),
        "delta_pp": delta,
        "ci_pp": [lo, hi],
        "se_pp": se,
        "p_permutation": stratified_permutation(n, sa, sb, draws, seed),
        "per_suite": per_suite,
        "per_key": [
            {"suite": k[0], "key": k[1], "episodes": int(n[i]), "successes_a": int(sa[i]), "successes_b": int(sb[i])}
            for i, k in enumerate(keys)
        ],
    }


def judge(record, min_delta):
    """Gate row: does `record` clear `min_delta` points with the CI clear of zero?"""
    lo = record["ci_pp"][0]
    return {"min_delta_pp": min_delta, "delta_pp": record["delta_pp"], "ci_low_pp": lo, "pass": bool(
        record["delta_pp"] >= min_delta and lo > 0.0
    )}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run", action="append", required=True, metavar="NAME=ROOT", help="repeatable; ROOT holds logs/<suite>/*.json"
    )
    parser.add_argument("--suites", default=None, help="comma list; default: every suite present in the first run")
    parser.add_argument(
        "--shard-glob",
        default="*.json",
        help="shard filename pattern per suite; LIBERO-Plus use '*_to_*.json' (default: *.json)",
    )
    parser.add_argument(
        "--skip-glob",
        action="append",
        default=None,
        metavar="PATTERN",
        help="repeatable; shard files to ignore, e.g. '*_to_*' to score the LIBERO endpoint only",
    )
    parser.add_argument(
        "--expected-per-suite",
        type=int,
        default=None,
        help="hard-fail unless every suite totals this many episodes (LIBERO: 500)",
    )
    parser.add_argument("--primary", default=None, help="arm the gate is about (default: no gate verdict)")
    parser.add_argument(
        "--reference", action="append", default=None, metavar="ARM", help="repeatable; every reference must be cleared"
    )
    parser.add_argument("--min-delta", type=float, default=3.0, help="gate margin in percentage points")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--list-shards", action="store_true", help="print every suite file with its episode total and exit"
    )
    args = parser.parse_args()

    runs = []
    for item in args.run:
        if "=" not in item:
            raise SystemExit(f"--run expects NAME=ROOT, got {item!r}")
        name, root = item.split("=", 1)
        runs.append((name, root))
    if len(runs) < 2 and not args.list_shards:
        raise SystemExit("need at least two --run entries to compare")
    if args.list_shards:
        for _, root in runs:
            list_shards(root)
        return
    if args.out_dir is None:
        raise SystemExit("--out-dir is required")
    if args.primary is not None and args.primary not in dict(runs):
        raise SystemExit(f"--primary {args.primary!r} is not one of {[n for n, _ in runs]}")

    suites = [s.strip() for s in args.suites.split(",") if s.strip()] if args.suites else None
    strata = {
        name: load_arm(root, suites, args.shard_glob, args.skip_glob or (), args.expected_per_suite)
        for name, root in runs
    }
    resolved = sorted({k[0] for k in next(iter(strata.values()))})
    print(f"[paired] suites: {','.join(resolved)}  draws={args.draws} seed={args.seed}")
    print(f"{'arm':<14} {'episodes':>9} {'successes':>10} {'SR':>8}")
    for name, _ in runs:
        total_n = sum(n for n, _ in strata[name].values())
        total_s = sum(s for _, s in strata[name].values())
        print(f"{name:<14} {total_n:>9} {total_s:>10} {total_s / total_n:>8.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = {}
    rows = []
    names = [name for name, _ in runs]
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            record = compare(strata[name_a], strata[name_b], name_a, name_b, args.draws, args.seed)
            pairs[f"{name_a}|{name_b}"] = record
            rows.append((name_a, name_b, record))
    for name_a, name_b, record in sorted(rows, key=lambda r: -abs(r[2]["delta_pp"])):
        lo, hi = record["ci_pp"]
        print(
            f"[paired] {name_a} vs {name_b:<14} d={record['delta_pp']:+7.3f}pp  "
            f"CI[{lo:+7.3f},{hi:+7.3f}]  SE={record['se_pp']:.3f}  p={record['p_permutation']:.4f}"
        )

    payload = {
        "suites": resolved,
        "draws": args.draws,
        "seed": args.seed,
        "min_delta_pp": args.min_delta,
        "arms": {
            name: {
                "episodes": int(sum(n for n, _ in strata[name].values())),
                "successes": int(sum(s for _, s in strata[name].values())),
            }
            for name, _ in runs
        },
        "pairs": pairs,
    }

    if args.primary is not None:
        references = args.reference or [n for n, _ in runs if n != args.primary]
        gate = {}
        for reference in references:
            if reference not in dict(runs):
                raise SystemExit(f"--reference {reference!r} is not one of {[n for n, _ in runs]}")
            key = f"{args.primary}|{reference}"
            if key not in pairs:
                # Recompute rather than negate: every field (per_suite sr_a/sr_b, per_key) then
                # stays self-consistent instead of describing the reverse comparison.
                pairs[key] = compare(
                    strata[args.primary], strata[reference], args.primary, reference, args.draws, args.seed
                )
            gate[reference] = judge(pairs[key], args.min_delta)
        verdict = all(row["pass"] for row in gate.values())
        print(f"\n[gate] margin +{args.min_delta:g}pp and CI low > 0 versus every reference")
        for reference, row in gate.items():
            mark = "PASS" if row["pass"] else "FAIL"
            print(
                f"[gate] {args.primary} vs {reference:<12} d={row['delta_pp']:+7.3f}pp  "
                f"CI low={row['ci_low_pp']:+7.3f}pp  {mark}"
            )
        print(f"[gate] VERDICT: {'GO' if verdict else 'NO-GO'}")
        payload["gate"] = {"primary": args.primary, "references": gate, "verdict": "GO" if verdict else "NO-GO"}

    (out_dir / "paired_eval.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[paired] wrote {out_dir / 'paired_eval.json'}")


if __name__ == "__main__":
    main()
