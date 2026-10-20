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
  shuffled    G rows permuted within the suite at cache level (pairing control).

Invalid transitions (NaN G rows, unmatched episodes) drop out of the OE term
only — the sample stays in the batch with the base loss untouched, keeping
the sample stream identical across all six arms.
"""

import numpy as np
import torch

OE_MODES = ("task_fixed", "trace", "diag", "full", "shuffled")


def oe_quadratic(payload, e_norm, mode, chunk_len):
    """Return (oe_loss, valid_frac) for one microbatch.

    payload: per-sample dicts {G, valid, half_range, task_mean, trace_mean};
    e_norm: [B, chunk_len * action_dim] normalized action error.
    """
    e_raw = e_norm * torch.as_tensor(
        np.stack([np.tile(np.asarray(ex["half_range"], dtype=np.float32), chunk_len) for ex in payload]),
        device=e_norm.device,
    )
    G_raw = torch.as_tensor(
        np.stack([np.asarray(ex["G"], dtype=np.float32) for ex in payload]), device=e_norm.device
    )  # [B, d, d] raw, NaN rows mark invalid transitions
    valid = torch.as_tensor(np.array([ex["valid"] for ex in payload], dtype=bool), device=e_norm.device)  # [B]
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
    quad = torch.einsum("bi,bij,bj->b", e_raw, G_eff, e_raw)
    quad = torch.where(valid, quad, torch.zeros_like(quad))
    oe_loss = quad.sum() / valid.float().sum().clamp(min=1.0)
    return oe_loss, valid.float().mean().detach()
