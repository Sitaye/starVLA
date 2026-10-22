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
"""OE-VLA tooling: outcome-geometry estimation, gates A/B, and policy instruments.

Gate A (zero training): replay-based counterfactual geometry, per-state
estimation and protocol checks -- docs/research/oevla_gate_a_protocol_20261019.md.
Gate B (training): offline G precompute, the OE auxiliary loss and its dose
calibration -- docs/research/oevla_gate_b_v2_design_20261020.md.
Instruments: policy_io (build/load), probe_policy_dims, calibrate_oe_lambda,
paired_eval (the gate's decision statistic over two eval trees).
Simulator work runs on the LIBERO stack; estimation/analysis are pure numpy and
unit tested on dev machines.
"""

__version__ = "0.1"

SEED = 42
