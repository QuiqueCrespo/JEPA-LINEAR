"""Fresh-sample training stream (module level so DataLoader workers can pickle it)."""
import numpy as np
import torch
from cpbench.data.dataset import Dataset


class Stream(torch.utils.data.IterableDataset):
    """Infinite fresh (context, target) batches; each worker owns an independent generator."""
    def __init__(self, spec, batch, window, gap, seed, start):
        self.args = (spec, batch, window, gap, seed, start)

    def __iter__(self):
        spec, batch, window, gap, seed, start = self.args
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        task = Dataset(spec)
        rng = np.random.default_rng([seed, 1000 + wid, start])
        while True:
            xc, xt, _ = task.train_batch(batch, window, gap, rng)
            yield torch.from_numpy(np.ascontiguousarray(xc)), torch.from_numpy(np.ascontiguousarray(xt))
