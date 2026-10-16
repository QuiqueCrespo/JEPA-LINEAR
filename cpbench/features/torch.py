"""Learnt feature learners: a shared PyTorch training loop.  Subclasses only implement build(D, W), returning a
module with  loss(xc, xt) -> (loss, dict of logged floats)  and  encode(x) -> (B, d)  (xc, xt: context / target
windows (B, W, D), the target W + gap samples after the context).  The loop handles fresh batches (new series every
step: no sample is ever seen twice), warm-up and an optional cosine decay, gradient clipping, skipping of non-finite
steps, resumable checkpoints, jsonl logs and periodic validation.

Periodic validation (every eval_every steps, on the fixed validation split) logs, as one "eval" record:
  supervised   probe_linear, probe_mlp_pair (cpbench/eval/probes.py; diagnostics, labels used)
  unsupervised acc / cp_ap of every detector in eval_detectors, scored at sample level like the test protocol
  geometry     eff_rank (participation ratio of the feature covariance), collapsed (eigenvalues < 1e-3 max)
keep_ckpts: steps whose weights are kept (ckpt_<step>.pt) for later test-set evaluation along training.
Long runs on SLURM: on SIGUSR1 / SIGTERM (sbatch --signal) the loop writes its checkpoint and exits with code 3,
which the sweep's job script turns into a requeue; the next run resumes from the checkpoint."""
import json, math, os, signal, time
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from cpbench.data.windows import majority_labels
from cpbench.detectors import make_detector
from cpbench.eval.probes import probes as run_probes
from cpbench.eval.scoring import score
from .base import FeatureLearner
from .stream import Stream


class TorchFeatureLearner(FeatureLearner):
    defaults = dict(window=32, gap=32, steps=20000, batch=256, lr=3e-4, warmup=1000, weight_decay=1e-4, clip=5.0,
                    schedule="constant", min_lr=0.0,
                    d=124, H=256, backbone="tcn", workers=4, device_name="auto", log_every=100, eval_every=2000,
                    ckpt_every=2000, eval_detectors=["gmm"], eval_probes="linear", keep_ckpts=[], seed=0)
    RUNTIME = ("workers", "device_name", "log_every", "eval_every", "ckpt_every", "eval_detectors", "eval_probes", "keep_ckpts")
    LEGACY = dict(schedule="constant", min_lr=0.0)   # left out of the run identity at these values (older runs keep theirs)
    TRAIN_KEYS = ("window", "gap", "steps", "batch", "lr", "warmup", "weight_decay", "clip", "schedule", "min_lr", "seed") + RUNTIME
    trainable = True
    EXIT_REQUEUE = 3

    def identity(self):
        return {k: v for k, v in super().identity().items() if not (k in self.LEGACY and v == self.LEGACY[k])}

    def lr_at(self, step):
        """Linear warm-up, then constant or cosine decay to min_lr * lr at the last step."""
        hp = self.params
        lr = hp["lr"] * min(1.0, (step + 1) / hp["warmup"])
        if hp["schedule"] == "cosine" and step >= hp["warmup"]:
            prog = (step - hp["warmup"]) / max(1, hp["steps"] - hp["warmup"])
            lr = hp["lr"] * (hp["min_lr"] + (1 - hp["min_lr"]) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))
        return lr

    def build(self, D, W):
        raise NotImplementedError

    def net_hp(self):
        """Parameters passed to the network (everything that is not a training-loop setting)."""
        return {k: v for k, v in self.params.items() if k not in self.TRAIN_KEYS}

    # ------------------------------------------------------------------ training
    def _device(self):
        name = self.device_name
        if name != "auto":
            return torch.device(name)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit(self, dataset, run_dir=None, log=print):
        hp, task = self.params, dataset
        torch.manual_seed(hp["seed"]); np.random.seed(hp["seed"])
        self.device = self._device()
        self.net = self.build(task.D, self.window).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        step, ck = 0, (os.path.join(run_dir, "ckpt.pt") if run_dir else None)
        if ck and os.path.exists(ck):
            st = torch.load(ck, map_location=self.device, weights_only=False)
            self.net.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); step = st["step"]
            log(f"resumed {run_dir} at step {step}")
        if step >= hp["steps"]:
            self.net.eval(); return self
        logf = open(os.path.join(run_dir, "log.jsonl"), "a") if run_dir else None   # closed at the end of fit
        def write(rec):
            if logf: logf.write(json.dumps(rec) + "\n"); logf.flush()
        val = task.val_set()
        loader = iter(torch.utils.data.DataLoader(Stream(task.config(), hp["batch"], self.window, hp["gap"], hp["seed"], step),
                                                  batch_size=None, num_workers=hp["workers"],
                                                  persistent_workers=hp["workers"] > 0,
                                                  prefetch_factor=4 if hp["workers"] > 0 else None))
        t0, acc, skipped, last_eval = time.time(), {}, 0, None
        stop = []                                                        # set by SLURM's pre-timeout signal
        handlers = {sig: signal.signal(sig, lambda *_: stop.append(1)) for sig in (signal.SIGUSR1, signal.SIGTERM)} if ck else {}
        evaluated = {r["step"] for r in map(json.loads, open(os.path.join(run_dir, "log.jsonl"))) if r.get("kind") == "eval"} if run_dir else set()
        self.net.train()
        while step < hp["steps"]:
            if stop:
                torch.save(dict(model=self.net.state_dict(), opt=opt.state_dict(), step=step), ck)
                write(dict(step=step, kind="requeue", t=time.time() - t0)); logf.close()
                log(f"[{self.name}] stopped by signal at step {step}; checkpoint written, exiting for requeue")
                raise SystemExit(self.EXIT_REQUEUE)
            if step % hp["eval_every"] == 0 and step != last_eval and step not in evaluated:   # skipped / resumed steps do not re-validate
                last_eval = step
                ev = self.validate(val, task.K, task.margin); write(dict(step=step, kind="eval", t=time.time() - t0, **ev))
                log(f"[{self.name} {task.info.id}] step {step:6d} {time.time()-t0:6.0f}s " + " ".join(f"{k} {v:.3f}" for k, v in ev.items()))
            if run_dir and step in hp["keep_ckpts"] and not os.path.exists(os.path.join(run_dir, f"ckpt_{step}.pt")):
                torch.save(dict(model=self.net.state_dict(), step=step), os.path.join(run_dir, f"ckpt_{step}.pt"))
            xc, xt = next(loader)
            nb = self.device.type == "cuda"                              # async copies from unpinned memory are unsafe on MPS
            xc, xt = xc.to(self.device, non_blocking=nb), xt.to(self.device, non_blocking=nb)
            for g in opt.param_groups:
                g["lr"] = self.lr_at(step)
            loss, parts = self.net.loss(xc, xt)
            if not torch.isfinite(loss):                                 # rare MPS forward NaNs: skip, log, fail if persistent
                skipped += 1; write(dict(step=step, kind="skip", msg="non-finite loss", **parts))
                if skipped > 20:
                    raise FloatingPointError(f"{skipped} non-finite losses by step {step}: {parts}")
                continue
            opt.zero_grad(set_to_none=True); loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), hp["clip"])
            if not torch.isfinite(gnorm):                                 # never let one bad gradient poison the weights
                skipped += 1; write(dict(step=step, kind="skip", msg="non-finite gradient"))
                if skipped > 20:
                    raise FloatingPointError(f"{skipped} non-finite gradients by step {step}")
                continue
            opt.step()
            for k, v in dict(loss=loss.item(), **parts).items():
                acc[k] = acc.get(k, 0.0) + v
            step += 1
            if step % hp["log_every"] == 0:
                write(dict(step=step, kind="train", t=time.time() - t0, lr=self.lr_at(step - 1),
                           **{k: v / hp["log_every"] for k, v in acc.items()})); acc = {}
            if ck and (step % hp["ckpt_every"] == 0 or step == hp["steps"]):
                torch.save(dict(model=self.net.state_dict(), opt=opt.state_dict(), step=step), ck)
        if step not in evaluated:
            ev = self.validate(val, task.K, task.margin); write(dict(step=step, kind="eval", t=time.time() - t0, **ev))
            log(f"[{self.name} {task.info.id}] done in {time.time()-t0:.0f}s: " + " ".join(f"{k} {v:.3f}" for k, v in ev.items()))
        if run_dir and step in hp["keep_ckpts"]:
            torch.save(dict(model=self.net.state_dict(), step=step), os.path.join(run_dir, f"ckpt_{step}.pt"))
        for sig, h in handlers.items():
            signal.signal(sig, h)
        if logf:
            logf.close()
        self.net.eval()
        return self

    def validate(self, val, K, margin=32):
        """Periodic check on the validation series: supervised probes (eval_probes: "linear" or "all", diagnostics),
        each eval_detectors detector's acc / cp_ap at sample level, collapse indicators.  Key names: probe (linear,
        kept for older logs), probe_mlp_pair, <detector>_acc, <detector>_cp_ap, eff_rank, collapsed."""
        self.net.eval()
        rep, starts = self.represent(val); Z = rep["features"].astype(np.float64)
        self.net.train()
        S = majority_labels(val["S_s"], starts, self.window, K); h = S.shape[0] // 2
        zf = Z.reshape(-1, Z.shape[-1])
        ev = np.atleast_1d(np.linalg.eigvalsh(np.atleast_2d(np.cov(zf.T))))
        out = dict(collapsed=int((ev < 1e-3 * ev.max()).sum()), eff_rank=float(ev.sum() ** 2 / (ev ** 2).sum()))
        if self.eval_probes == "all":
            pr = run_probes(Z, S); out.update(probe=float(pr["probe_linear"]), probe_mlp_pair=float(pr["probe_mlp_pair"]))
        else:
            zs = (zf - zf.mean(0)) / (zf.std(0) + 1e-9); n_tr = h * S.shape[1]
            out["probe"] = float(LogisticRegression(max_iter=2000, C=0.1).fit(zs[:n_tr], S[:h].ravel()).score(zs[n_tr:], S[h:].ravel()))
        for name in self.eval_detectors:
            sc = score(make_detector(name).detect(Z, K, seed=0), starts, self.window, val, K, margin)
            out.update({f"{name}_{m}": float(sc[m]) for m in ("acc", "cp_ap") if m in sc})
        return out

    # ------------------------------------------------------------------ inference / persistence
    def _encode(self, windows, fn):
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32)).to(self.device)
            return fn(x).float().cpu().numpy()

    def embed(self, windows):
        return self._encode(windows, self.net.encode)

    def save(self, path):
        torch.save(dict(model=self.net.state_dict(), params=self.identity()), path)

    def load(self, path, D=None):
        st = torch.load(path, map_location="cpu", weights_only=False)
        self.device = self._device()
        if not hasattr(self, "net"):
            self.net = self.build(D, self.window)
        self.net.load_state_dict(st["model"]); self.net.to(self.device).eval()
        return self
