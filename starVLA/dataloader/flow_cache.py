import numpy as np


class FlowTargetCache:
    """
    Read-only view over a precomputed RAFT flow-target cache, one directory per dataset:
    ``<root>/<dataset_name>/flow.npy`` ([N, V, H_tok, W_tok, 2] float16) and
    ``keys.npy`` ([N] int64 sorted, key = trajectory_id * 100_000 + step).
    Backed by memmaps so forked DataLoader workers share pages without copying.
    """

    def __init__(self, root: str):
        self.root = root
        self._entries = {}

    def _load(self, dataset_name: str):
        if dataset_name not in self._entries:
            directory = f"{self.root}/{dataset_name}"
            flow = np.load(f"{directory}/flow.npy", mmap_mode="r")
            keys = np.load(f"{directory}/keys.npy")
            self._entries[dataset_name] = (flow, keys)
        return self._entries[dataset_name]

    def get(self, dataset_name: str, trajectory_id: int, step: int) -> np.ndarray:
        """Return the [V, H_tok, W_tok, 2] float16 flow target for one step."""
        flow, keys = self._load(dataset_name)
        key = trajectory_id * 100_000 + step
        index = int(np.searchsorted(keys, key))
        if index >= len(keys) or keys[index] != key:
            raise KeyError(f"flow target not found for {dataset_name} trajectory {trajectory_id} step {step}")
        return np.asarray(flow[index])
