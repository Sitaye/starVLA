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
"""OE-VLA gate A: counterfactual outcome-geometry validation package.

Pure-simulation, zero-training validation of the OE-VLA main hypothesis
(docs/research/ac_review_new_idea_oevla_20261019.md). Protocol:
docs/research/oevla_gate_a_protocol_20261019.md. Cluster-only execution
(LIBERO/robosuite); estimation and analysis layers are pure numpy and unit
tested on dev machines.
"""

__version__ = "0.1"

SEED = 42
