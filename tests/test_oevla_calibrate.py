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
"""Tests for the gate-B lambda calibration instrument."""

import numpy as np
import pytest
import torch

from starVLA.analysis.oevla import calibrate_oe_lambda as CAL


def test_calibrate_equalises_gradient_strength():
    ratios = {"taskfixed": 2.0, "trace_eef": 0.5, "full": 8.0}
    lambdas = CAL.calibrate(ratios, "taskfixed", 0.56)
    # the anchor keeps its historical lambda, everybody else lands on the same rho
    assert lambdas["taskfixed"] == pytest.approx(0.56)
    for arm, lam in lambdas.items():
        assert lam * ratios[arm] == pytest.approx(0.56 * ratios["taskfixed"])
    # a metric with a larger unit gradient needs proportionally less weight
    assert lambdas["full"] < lambdas["taskfixed"] < lambdas["trace_eef"]


class _SumOfTwoTerms(torch.nn.Module):
    """L_BC = |w|_1 (grad = sign(w)), L_OE = |w|_2^2 (grad = 2w); both closed-form."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.oe_loss_weight = 0.0
        self.oe_mode = None

    def forward(self, examples=None):
        base = self.w.abs().sum()
        aux = self.w.square().sum()
        return {"total_loss": base + self.oe_loss_weight * aux}


def _oe_payload(valid=True, g=None):
    """Minimal per-sample OE payload; batch element shape follows the real attach contract."""
    g = np.full((3, 3), np.nan) if not valid else (np.eye(3) if g is None else np.asarray(g))
    return {"G": g, "valid": valid, "half_range": np.ones(2), "task_mean": np.eye(3), "trace_mean": 1.0}


def _batch(*valid):
    return [{"qwen_inputs": {}, "oe": _oe_payload(v)} for v in (valid or (True,))]


def test_flat_grad_counts_unused_params_as_zero():
    model = _SumOfTwoTerms()
    unused = torch.nn.Parameter(torch.tensor([5.0]))
    g = CAL._flat_grad(model()["total_loss"], [model.w, unused])
    np.testing.assert_allclose(g, [1.0, 1.0, 0.0], atol=1e-6)


def test_gradients_recover_aux_from_toggled_objective():
    model = _SumOfTwoTerms()
    model.oe_loss_weight = 0.25
    g_bc, g_oe = CAL.oe_gradients(model, None, list(model.parameters()))
    np.testing.assert_allclose(g_bc, [1.0, 1.0], atol=1e-6)  # grad |w|_1 = sign(w)
    np.testing.assert_allclose(g_oe, [2.0, 4.0], atol=1e-6)  # grad |w|_2^2 = 2w
    assert model.oe_loss_weight == 0.25  # the arm's own weight is restored


class _NaNOnAux(torch.nn.Module):
    """Base term is fine; the auxiliary term's gradient is NaN (the poison we guard against)."""
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.oe_loss_weight = 0.0
        self.oe_mode = "full"

    def forward(self, examples=None):
        aux = self.w.square().sum() if self.oe_loss_weight == 0.0 else self.w.square().sum() * torch.tensor(float("nan"))
        return {"total_loss": self.w.abs().sum() + self.oe_loss_weight * aux}


def test_gradient_ratio_fails_loud_on_nan_aux():
    model = _NaNOnAux()
    with pytest.raises(RuntimeError, match="non-finite gradient"):
        CAL.oe_gradients(model, None, list(model.parameters()))


def test_measure_arm_takes_the_ratio_of_the_step_gradient():
    """Identical batches: summing then dividing equals the single-batch ratio."""
    model = _SumOfTwoTerms()
    loader = [_batch()] * 4  # two windows of two microbatches
    ratios, invalid, g_max = CAL.measure_arm(model, loader, batches=2, windows=2, params=list(model.parameters()))
    assert len(ratios) == 2  # two disjoint one-step windows
    assert np.mean(ratios) == pytest.approx(np.sqrt(20) / np.sqrt(2), rel=1e-6)
    assert invalid == 0.0 and g_max == pytest.approx(1.0)


def test_measure_arm_cancels_incoherent_aux_noise():
    """The trainer steps the *summed* gradient: per-microbatch OE noise must cancel.

    Averaging per-microbatch ratios would report ~2.0 here even though the OE term
    contributes nothing to the step, which is how the noisy arms got under-dosed.
    """

    class _Alternating(_SumOfTwoTerms):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, examples=None):
            phase = -1.0 if (self.calls // 2) % 2 else 1.0  # one phase per microbatch
            self.calls += 1
            return {"total_loss": self.w.abs().sum() + self.oe_loss_weight * phase * self.w.square().sum()}

    model = _Alternating()
    ratios, _, _ = CAL.measure_arm(model, [_batch(), _batch()], batches=2, windows=1, params=list(model.parameters()))
    assert ratios[0] < 1e-6


def test_measure_arm_respects_batch_budget():
    class _Counter(_SumOfTwoTerms):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, examples=None):
            self.calls += 1
            return super().forward()

    model = _Counter()
    CAL.measure_arm(model, [_batch()] * 5, batches=2, windows=2, params=list(model.parameters()))
    assert model.calls == 8  # 2 weight passes x 2 microbatches x 2 windows


def test_measure_arm_rejects_empty_stream():
    model = _SumOfTwoTerms()
    with pytest.raises(RuntimeError, match="no batches"):
        CAL.measure_arm(model, [], batches=2, windows=2, params=list(model.parameters()))


def test_batch_stats_report_the_invalid_fraction():
    """Tail steps (base_index in the last chunk-1) arrive invalid; the arm must say so."""
    invalid, g_max = CAL.oe_batch_stats(_batch(True, False, True, False))
    assert invalid == pytest.approx(0.5)
    assert g_max == pytest.approx(1.0)
    all_bad, g_bad = CAL.oe_batch_stats(_batch(False, False))
    assert all_bad == 1.0 and np.isnan(g_bad)
