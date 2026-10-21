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


def test_flat_grad_counts_unused_params_as_zero():
    model = _SumOfTwoTerms()
    unused = torch.nn.Parameter(torch.tensor([5.0]))
    g = CAL._flat_grad(model()["total_loss"], [model.w, unused])
    np.testing.assert_allclose(g, [1.0, 1.0, 0.0], atol=1e-6)


def test_gradient_ratio_recovers_aux_from_toggled_objective():
    model = _SumOfTwoTerms()
    ratio = CAL.oe_gradient_ratio(model, None, list(model.parameters()))
    # ||grad L_OE|| = ||(2, 4)|| = sqrt(20); ||grad L_BC|| = ||(1, 1)|| = sqrt(2)
    assert ratio == pytest.approx(np.sqrt(20) / np.sqrt(2), rel=1e-6)
    assert model.oe_loss_weight == 1.0  # left enabled after the measurement pass


def test_measure_arm_sets_mode_and_averages_batches():
    model = _SumOfTwoTerms()
    loader = [[{"qwen_inputs": {}}], [{"qwen_inputs": {}}]]
    ratios = CAL.measure_arm(model, loader, "diag", batches=2, params=list(model.parameters()))
    assert model.oe_mode == "diag"
    assert len(ratios) == 2
    assert ratios[0] == pytest.approx(ratios[1])


def test_measure_arm_respects_batch_budget():
    model = _SumOfTwoTerms()
    loader = [[{"qwen_inputs": {}}]] * 5
    ratios = CAL.measure_arm(model, loader, "full", batches=2, params=list(model.parameters()))
    assert len(ratios) == 2


def test_measure_arm_rejects_empty_stream():
    model = _SumOfTwoTerms()
    with pytest.raises(RuntimeError, match="no batches"):
        CAL.measure_arm(model, [], "full", batches=2, params=list(model.parameters()))
