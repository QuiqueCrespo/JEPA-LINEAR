"""Fresh-sample training stream (module level so DataLoader workers can pickle it)."""
import numpy as np
import torch
from cpbench.data.dataset import Dataset


class Stream(torch.utils.data.IterableDataset):
    """Infinite fresh (context, target) batches; batch-indexed randomness is independent of worker count and resumes exactly."""
    def __init__(self, spec, batch, window, gap, seed, start, version=2):
        self.args = (spec, batch, window, gap, seed, start, version)

    def __iter__(self):
        spec, batch, window, gap, seed, start, version = self.args
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        task = Dataset(spec)
        if version != 2:
            raise ValueError("Only stream_version=2 is supported; use the historical code for legacy streams")
        batch_index = start + wid
        stride = info.num_workers if info else 1
        while True:
            rng = np.random.default_rng([seed, 1000, batch_index])
            xc, xt, _ = task.train_batch(batch, window, gap, rng)
            batch_index += stride
            yield torch.from_numpy(np.ascontiguousarray(xc)), torch.from_numpy(np.ascontiguousarray(xt))
