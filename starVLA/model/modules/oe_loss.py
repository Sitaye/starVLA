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
"""Gate-B outcome-equivalence (OE) auxiliary loss.

Pre-registered protocol v2.0.1 (docs/research/oevla_gate_b_v2_design_20261020.md
+ gate A adjudication): the action error is rescaled to raw units by the
per-dim half_range and quadratically reweighted by each transition's
precomputed outcome geometry. All G-family arms are unit-mean-diagonal, so
arm differences are purely structural:

  task_fixed  suite-mean of the per-state normalized G (state-free control);
  trace       (mean-diag / suite-mean) * I — SCRIM-style state scalar;
  diag        diagonal of the per-state normalized G;
  full        per-state normalized G (the OE-VLA main arm);
  shuffled    G rows permuted within the suite at cache level (pairing control);
  trace_eef   identity on the EEF cells only, gripper masked (v2 null).

`trace` applies the raw-unit identity and is therefore dominated by the gripper
channel (half_range_grip = 1.0 vs ~0.03 for the rotation dims), while every G-family
arm carries an exactly zero gripper weight — the analytic Jacobian is blind to the
binary channel. It is the demonstration arm for that failure, not a null; the null
is `trace_eef`, which masks the gripper so all geometry-bearing arms share the same
cell support and a mean diagonal of 1.

Every mode is normalized to a **mean over cells** (not a sum): the base loss is an L1
mean over the same cells, so λ stays a dimensionless weight relative to it, and stays
comparable across chunk lengths. A sum would make the auxiliary term D times larger
than its nominal λ, which is how the first three rounds stayed at or above parity with
the base loss even at λ = 0.01.

The convention fixes the units, not the magnitude: ``trace`` runs in raw units with the
per-state scale kept (~half_range² times the unit-mean arms) and ``trace_eef`` runs in
normalized units. λ is therefore **not** numerically comparable across modes; each arm is
dosed by gradient calibration (design doc 6.10), which equalizes the achieved dose
instead of the nominal weight.

Invalid transitions (NaN G rows, unmatched episodes) drop out of the OE term
only — the sample stays in the batch with the base loss untouched, keeping
the sample stream identical across all arms.
"""

import numpy as np
import torch

OE_MODES = ("task_fixed", "trace", "diag", "full", "shuffled", "trace_eef")
GRIPPER_IDX = 6  # binary channel; masked by trace_eef, zero-weighted by every G-family arm


def oe_quadratic(payload, e_norm, mode, chunk_len):
    """Return (oe_loss, valid_frac) for one microbatch.

    payload: per-sample dicts {G, valid, half_range, task_mean, trace_mean};
    e_norm: [B, chunk_len * action_dim] normalized action error.
    """
    valid = torch.as_tensor(np.array([ex["valid"] for ex in payload], dtype=bool), device=e_norm.device)  # [B]
    d_total = e_norm.shape[-1]
    if mode == "trace_eef":
        # The v2 null lives in the normalized action space and reads no geometry at all.
        if d_total % chunk_len:
            raise ValueError(f"action error width {d_total} is not a multiple of chunk_len {chunk_len}")
        keep = torch.arange(d_total, device=e_norm.device) % (d_total // chunk_len) != GRIPPER_IDX
        # Same cell count as the G-family arms (gripper cells contribute exact zero), so the
        # mean-over-cells convention is identical across every mode.
        quad = e_norm.square()[:, keep].sum(-1) / d_total
    else:
        e_raw = e_norm * torch.as_tensor(
            np.stack([np.tile(np.asarray(ex["half_range"], dtype=np.float32), chunk_len) for ex in payload]),
            device=e_norm.device,
        )
        G_raw = torch.as_tensor(
            np.stack([np.asarray(ex["G"], dtype=np.float32) for ex in payload]), device=e_norm.device
        )  # [B, d, d] raw, NaN rows mark invalid transitions
        # Invalid rows are masked at the end, but torch.where only zeroes the value and the
        # upstream gradient: the VJP of the quadratic form still multiplies by G, and
        # 0 * NaN = NaN. An unmasked NaN row therefore poisons the whole parameter gradient
        # while the reported loss stays finite. Zero the rows before any arithmetic.
        G_raw = torch.nan_to_num(G_raw, nan=0.0, posinf=0.0, neginf=0.0)
        if mode == "task_fixed":
            G_eff = torch.as_tensor(
                np.stack([np.asarray(ex["task_mean"], dtype=np.float32) for ex in payload]), device=e_norm.device
            )
        elif mode == "trace":
            c = torch.diagonal(G_raw, dim1=-2, dim2=-1).mean(-1)  # [B]
            w = c / c.new_tensor(np.array([ex["trace_mean"] for ex in payload])).clamp(min=1e-12)
            G_eff = torch.eye(G_raw.shape[-1], device=e_norm.device).unsqueeze(0) * w[:, None, None]
        elif mode == "diag":
            diag = torch.diagonal(G_raw, dim1=-2, dim2=-1)
            G_eff = torch.diag_embed(diag / diag.mean(-1, keepdim=True).clamp(min=1e-12))
        else:  # full / shuffled
            c = torch.diagonal(G_raw, dim1=-2, dim2=-1).mean(-1, keepdim=True).clamp(min=1e-12)  # [B, 1]
            G_eff = G_raw / c.unsqueeze(-1)  # [B, d, d]
        quad = torch.einsum("bi,bij,bj->b", e_raw, G_eff, e_raw) / d_total
    quad = torch.where(valid, quad, torch.zeros_like(quad))
    # Mean over the batch, not over the valid rows: a sample without a geometry row still
    # carries weight 1 in the base L1, so it must carry the same weight here (invalid rows
    # contribute exact zero). Dividing by n_valid instead would scale every arm's effective
    # dose by B/n_valid, i.e. by the dataset's episode-length distribution.
    oe_loss = quad.sum() / max(len(payload), 1)
    return oe_loss, valid.float().mean().detach()
