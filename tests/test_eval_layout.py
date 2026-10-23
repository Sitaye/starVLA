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
"""Guards the eval artifact layout: <ckpt>/<name>/eval/<endpoint>/{logs,videos,...}.

The two endpoints share every suite name and every driver wipes `logs/<suite>/*.json`
before launching, so the endpoint has to be a path *component*. When it was not, the
second endpoint to run on a checkpoint silently deleted the first one's shards.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

DRIVERS = {
    "libero": [
        "examples/simBenchmarks/LIBERO/eval_files/auto_eval_scripts/eval_libero_in_one.sh",
        "examples/simBenchmarks/LIBERO/eval_files/auto_eval_scripts/auto_eval_libero.sh",
        "examples/simBenchmarks/LIBERO/eval_files/auto_eval_scripts/eval_libero_parall.sh",
    ],
    "libero_plus": [
        "examples/simBenchmarks/LIBERO-plus/eval_files/eval_libero_in_one.sh",
        "examples/simBenchmarks/LIBERO-plus/eval_files/auto_eval_libero_plus.sh",
        "examples/simBenchmarks/LIBERO-plus/eval_files/eval_libero.sh",
    ],
}

IN_ONE = [DRIVERS["libero"][0], DRIVERS["libero_plus"][0]]


def _eval_roots(name):
    text = (REPO / name).read_text()
    tag = re.search(r'^\s*ENDPOINT="([^"]+)"', text, re.MULTILINE)
    lines = [line.strip() for line in text.splitlines() if re.match(r"\s*eval_root=", line)]
    return [line.replace("${ENDPOINT}", tag.group(1)) if tag else line for line in lines]


@pytest.mark.parametrize("endpoint", sorted(DRIVERS))
def test_every_driver_scopes_its_root_to_one_endpoint(endpoint):
    for name in DRIVERS[endpoint]:
        roots = _eval_roots(name)
        assert roots, f"{name} does not assign eval_root"
        for root in roots:
            assert f"/eval/{endpoint}" in root, f"{name}: {root} is not scoped to {endpoint}"


def test_the_endpoint_is_the_last_path_component():
    """A suffix on the checkpoint name (`<name>_plus`) would not be sufficient: `/eval/<endpoint>`
    is, because the two trees cannot produce the same final component."""
    for endpoint, names in DRIVERS.items():
        root = _eval_roots(names[0])[0]
        tail = root.rsplit("/eval/", 1)[1].strip('"')
        assert tail == endpoint, f"{names[0]}: root ends with {tail!r}, expected {endpoint!r}"
        assert tail not in {"", "..", "."}


@pytest.mark.parametrize("name", IN_ONE)
def test_the_wipe_stays_inside_the_endpoint_root(name):
    """`rm -f ${log_path}/*.json` is only safe while log_path derives from eval_root."""
    text = (REPO / name).read_text()
    assert 'log_path="${eval_root}/logs/${task_suite_name}"' in text
    assert 'rm -f "${log_path}"/*.json' in text
