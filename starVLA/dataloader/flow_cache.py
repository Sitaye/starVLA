import numpy as np


class FlowTargetCache:
    """
    Read-only view over a precomputed RAFT flow-target cache, one directory per dataset:
    ``<root>/<dataset_name>/flow.npy`` ([N, V, H_tok, W_tok, 2] float16),
    ``<root>/<dataset_name>/q.npy`` ([N, V, H_tok, W_tok] float16, motion confidence in
    [q_min, 1], see starVLA.model.modules.motion.head.flow_confidence_q) and
    ``keys.npy`` ([N] int64 sorted, key = trajectory_id * 100_000 + step).
    Backed by memmaps so forked DataLoader workers share pages without copying.
    Invalid anchors (step + delta_t >= trajectory length) stay NaN in both arrays.
    """

    def __init__(self, root: str):
        self.root = root
        self._entries = {}

    def _load(self, dataset_name: str):
        if dataset_name not in self._entries:
            directory = f"{self.root}/{dataset_name}"
            flow = np.load(f"{directory}/flow.npy", mmap_mode="r")
            q = np.load(f"{directory}/q.npy", mmap_mode="r")
            keys = np.load(f"{directory}/keys.npy")
            self._entries[dataset_name] = (flow, q, keys)
        return self._entries[dataset_name]

    def get(self, dataset_name: str, trajectory_id: int, step: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the ([V, H_tok, W_tok, 2] flow, [V, H_tok, W_tok] q) float16 targets for one step."""
        flow, q, keys = self._load(dataset_name)
        key = trajectory_id * 100_000 + step
        index = int(np.searchsorted(keys, key))
        if index >= len(keys) or keys[index] != key:
            raise KeyError(f"flow target not found for {dataset_name} trajectory {trajectory_id} step {step}")
        return np.asarray(flow[index]), np.asarray(q[index])
