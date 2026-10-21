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
"""Gate-B lambda calibration: give every arm the same auxiliary gradient strength.

lambda_OE was previously chosen by halving a decade at a time (1.0 -> 0.1 -> 0.01),
which is not a derivation: the arms' quadratics differ by orders of magnitude in raw
units, so one number cannot mean the same thing for two arms. This script measures the
quantity that actually competes with the base loss — the parameter-gradient ratio

    r_arm(lambda) = ||d L_OE / d theta|| / ||d L_BC / d theta||  =  lambda * r_arm(1)

since the two terms add, and solves lambda_arm = rho / r_arm(1) for a target ratio rho.

r_arm is measured on the *step* gradient: the per-microbatch gradients are summed over
the replay before the norms are taken, because training sums them too (8 ranks x 16
samples, ZeRO-2 with gradient_accumulation_steps 1). Averaging per-microbatch ratios
instead would inflate r by the incoherent part of the OE gradient — most for the noisiest
arm — and systematically under-dose the arms being compared.

The sample count matters for the same reason: the coherent part of the base gradient grows
~N while the incoherent part of the OE gradient grows ~sqrt(N), so r is only *the training
step's* ratio when N matches one step. With per_device_batch_size 16 and 8 ranks that is
128 samples, i.e. --batches 8 on a single process; scaling --batches scales r.
rho is anchored on the one arm with a positive closed-loop result (taskfixed, which at
the pre-normalisation lambda = 0.01 — i.e. 0.01 * 56 = 0.56 once the loss is a mean over
cells rather than a sum — reached 97.2%), so the re-run asks whether *structure* matters
at a dose that is known not to destroy the policy.

Zero training: a fixed sample stream, two forwards and two autograd.grad calls per
microbatch. Each arm gets its own framework built from its own YAML, because the arm is
just as much `datasets.vla_data.oe_shuffle` (the shuffled control) as it is
`framework.oe.mode`; measuring every arm on one dataloader would silently turn the
pairing control into a copy of `full`. Run it in the StarVLA training environment (torch, transformers,
accelerate, omegaconf, pyav); it never touches robosuite.

Usage (cluster, GPU):
    python -m starVLA.analysis.oevla.calibrate_oe_lambda \
        --ckpt /path/to/steps_10000_pytorch_model.pt \
        --arm taskfixed=examples/simBenchmarks/LIBERO/train_files/train_oeB_taskfixed.yaml \
        --arm trace_eef=.../train_oeB_trace_eef.yaml \
        --arm diag=.../train_oeB_diag.yaml \
        --arm full=.../train_oeB_full.yaml \
        --anchor-arm taskfixed --anchor-lambda 0.56 \
        --out-dir bar/oe_calibration
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from starVLA.analysis.oevla.policy_io import build_policy, check_preprocessed, load_checkpoint, to_device


def _flat_grad(loss, params):
    """Concatenate d loss / d params into one float64 vector (None grads count as 0)."""
    grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
    blocks = [
        np.zeros(p.numel(), dtype=np.float64) if g is None else g.detach().float().cpu().numpy().ravel()
        for p, g in zip(params, grads, strict=True)
    ]
    return np.concatenate(blocks)


def oe_gradients(model, examples, params):
    """(grad L_BC, grad L_OE) for one microbatch, as flat float64 vectors.

    The two terms are read off one additive objective by toggling the weight: the
    lambda = 1 pass yields grad(L_BC + L_OE) and the lambda = 0 pass grad(L_BC), so
    the difference is exactly grad(L_OE) even though the framework reports the OE
    term detached. The arm's own weight is restored afterwards.
    """
    keep = model.oe_loss_weight
    model.oe_loss_weight = 0.0
    g_bc = _flat_grad(model(examples=examples)["total_loss"], params)
    model.oe_loss_weight = 1.0
    g_total = _flat_grad(model(examples=examples)["total_loss"], params)
    model.oe_loss_weight = keep
    g_oe = g_total - g_bc
    n_bc, n_oe = int((~np.isfinite(g_bc)).sum()), int((~np.isfinite(g_oe)).sum())
    if n_bc or n_oe:
        # A NaN here used to surface as a silent `r=nan` row in the summary; say which
        # term broke and on what scale, so a stale cache or a bad mask cannot hide.
        raise RuntimeError(
            f"non-finite gradient with mode {model.oe_mode!r}: {n_bc} bad cells in the base term, "
            f"{n_oe} in the OE term (|g_bc|={np.linalg.norm(np.nan_to_num(g_bc)):.3g}, "
            f"|g_oe|={np.linalg.norm(np.nan_to_num(g_oe)):.3g}); inspect the payload G rows and the "
            "arm's metric"
        )
    return g_bc, g_oe


def calibrate(ratios, anchor_arm, anchor_lambda):
    """lambda per arm so that lambda_arm * r_arm equals the anchor's gradient ratio."""
    rho = float(anchor_lambda) * float(ratios[anchor_arm])
    return {arm: rho / r for arm, r in ratios.items()}


def oe_batch_stats(examples):
    """(invalid fraction, max |G|) for one microbatch -- the arm's own health report.

    Every training sample whose base_index sits in the last chunk-1 steps has no cache
    row and arrives invalid by design; a fraction near 1.0 instead means the cache and
    the sampler disagree about episode lengths.
    """
    payload = [ex["oe"] for ex in examples]
    invalid = float(np.mean([not ex["valid"] for ex in payload]))
    finite = [np.abs(np.asarray(ex["G"], dtype=np.float32)) for ex in payload if ex["valid"]]
    max_abs = float(np.max(np.concatenate([x.ravel() for x in finite]))) if finite else float("nan")
    return invalid, max_abs


def measure_arm(model, loader, batches, windows, params):
    """Replay one-step windows; return (per-window step ratios, invalid frac, max |G|).

    Each window sums ``batches`` microbatches, i.e. one training step's worth of samples, so
    every element is an estimate of the ratio the optimizer actually sees. The windows use
    disjoint parts of the deterministic stream, which is the only way to get an error bar:
    the stream is fixed per (epoch, index, seed), so two runs of the script return exactly
    the same number and a single window would hide its own sampling error.
    """
    device = next(model.parameters()).device
    stream = iter(loader)
    ratios, invalid, g_max = [], [], []
    for _ in range(windows):
        g_bc = g_oe = None
        for _ in range(batches):
            try:
                examples = next(stream)
            except StopIteration:
                break
            check_preprocessed(examples)
            sta = oe_batch_stats(examples)
            invalid.append(sta[0])
            g_max.append(sta[1])
            bc, oe = oe_gradients(model, to_device(examples, device), params)
            g_bc = bc if g_bc is None else g_bc + bc
            g_oe = oe if g_oe is None else g_oe + oe
        if g_bc is None:
            break
        ratios.append(float(np.linalg.norm(g_oe) / max(np.linalg.norm(g_bc), 1e-12)))
    if not ratios:
        raise RuntimeError("dataloader produced no batches")
    return ratios, float(np.mean(invalid)), float(np.nanmax(g_max))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="checkpoint whose gradient landscape defines the dose")
    parser.add_argument(
        "--arm", action="append", required=True, metavar="NAME=YAML", help="repeatable; the YAML's mode picks the metric"
    )
    parser.add_argument("--anchor-arm", default="taskfixed", help="arm whose historical dose defines the target ratio")
    parser.add_argument(
        "--anchor-lambda", type=float, default=0.56, help="that arm's historical lambda, mean-over-cells units"
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=8,
        help="microbatches per window; must add up to ONE training step (16 per rank x 8 ranks = 128 -> 8)",
    )
    parser.add_argument("--windows", type=int, default=4, help="disjoint one-step windows averaged for the error bar")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    arms = []
    for item in args.arm:
        if "=" not in item:
            raise SystemExit(f"--arm expects NAME=YAML, got {item!r}")
        name, path = item.split("=", 1)
        if not Path(path).is_file():
            raise SystemExit(f"{name}: arm config not found: {path}")
        arms.append((name, path))
    if args.anchor_arm not in dict(arms):
        raise SystemExit(f"--anchor-arm {args.anchor_arm!r} is not one of {[a for a, _ in arms]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modes, ratios, per_window_sem = {}, {}, {}
    for name, path in arms:
        model, loader = build_policy(path, out_dir, args.device)  # arm == its own YAML
        load_checkpoint(model, args.ckpt)
        params = [p for p in model.parameters() if p.requires_grad]
        modes[name] = model.oe_mode
        per_window, invalid, g_max = measure_arm(model, loader, args.batches, args.windows, params)
        ratios[name] = float(np.mean(per_window))
        spread = np.std(per_window, ddof=1) if len(per_window) > 1 else 0.0
        per_window_sem[name] = float(spread / np.sqrt(len(per_window)))
        print(
            f"[calib] {name:<12} mode={modes[name]:<11} r={ratios[name]:.4f} +/- {per_window_sem[name]:.4f}"
            f"  ({len(per_window)} step-window(s), invalid {invalid:.3f}, max|G| {g_max:.3g})"
        )
        del model, loader
        gc.collect()
        torch.cuda.empty_cache()  # the next arm builds its own framework

    lambdas = calibrate(ratios, args.anchor_arm, args.anchor_lambda)
    rho = args.anchor_lambda * ratios[args.anchor_arm]

    print(f"\n[calib] anchor {args.anchor_arm}: lambda {args.anchor_lambda:g} -> rho = {rho:.4f}")
    print(f"[calib] {args.batches} microbatch(es) per arm; r is a step ratio only if that is one training step")
    print(f"{'arm':<12} {'mode':<12} {'r=lambda1':>10} {'lambda*':>10} {'rho@lambda*':>12}")
    for name, _ in arms:
        print(
            f"{name:<12} {modes[name]:<12} {ratios[name]:>10.4f} {lambdas[name]:>10.4f} "
            f"{lambdas[name] * ratios[name]:>12.4f}"
        )

    payload = {
        "checkpoint": str(args.ckpt),
        "batches": args.batches,
        "windows": args.windows,
        "anchor_arm": args.anchor_arm,
        "anchor_lambda": args.anchor_lambda,
        "target_ratio": rho,
        "arms": {
            name: {
                "mode": modes[name],
                "grad_ratio_at_lambda_1": ratios[name],
                "grad_ratio_sem": per_window_sem[name],
                "lambda": lambdas[name],
            }
            for name, _ in arms
        },
    }
    (out_dir / "calibration.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[calib] wrote {out_dir / 'calibration.json'}")
    print("[calib] launch each arm with --framework.oe.loss_weight <lambda*> (do not edit the YAML)")


if __name__ == "__main__":
    main()
