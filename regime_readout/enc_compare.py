import os, sys, math, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, nbformat, pandas as pd
from dataclasses import dataclass
os.environ["OMP_NUM_THREADS"] = "4"; torch.set_num_threads(4)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
ns = dict(np=np, math=math, torch=torch, nn=nn, F=F, dataclass=dataclass, DEVICE=torch.device("cpu"), EMBED_DIM=124, HIDDEN=256,
          LogisticRegression=LogisticRegression, LinearRegression=LinearRegression, MLPClassifier=MLPClassifier, GaussianMixture=GaussianMixture,
          balanced_accuracy_score=balanced_accuracy_score, StandardScaler=StandardScaler, PCA=PCA, linear_sum_assignment=linear_sum_assignment, pd=pd)
# helper functions: evaluation + clustering cells of regime_large_scale.ipynb; model + generator from long_run.py
nb = nbformat.read("regime_large_scale.ipynb", 4)
for c in nb.cells:
    if c.cell_type == "code" and ("def cluster_battery" in c.source or "def predictable_variance" in c.source):
        exec(c.source, ns)
src = open("long_run.py").read()
for a, b in [("class SIGReg", "\ndef batch_whiten"), ("@dataclass\nclass BigSpec", "\ndef window_features"), ("class BigTCN", "\n# %%")]:
    i = src.index(a); j = src.find(b, i); chunk = src[i:j if j > 0 else len(src)]
    if a == "class BigTCN" and "EVAL_S" in chunk: chunk = chunk[: chunk.index("EVAL_S")]
    exec(chunk, ns)
spec = ns["BigSpec"](); ns["spec"] = spec
rng = np.random.default_rng(spec.seed)
_ = ns["simulate_big"](spec, 64, 2000, rng); test = ns["simulate_big"](spec, 16, 2000, rng)    # same test set as the run
torch.manual_seed(0); corr = ns["Corruptor"](spec)
with torch.no_grad():
    u = torch.tensor(_["u"][:8, :256]).reshape(-1, spec.W, spec.n_lat); lg = torch.tensor(_["logg"][:8, :256]).reshape(-1, spec.W)
    o = corr(u, lg, gen=torch.Generator().manual_seed(1)); corr.mu.copy_(o.mean((0, 1))); corr.sd.copy_(o.std((0, 1)) + 1e-6)
    SER = list(range(6))
    obs = torch.cat([corr(torch.tensor(test["u"][s]), torch.tensor(test["logg"][s]), gen=torch.Generator().manual_seed(10_000 + s)) for s in SER])
S = test["S"][SER]
m = ns["BigJEPA"]("separate"); ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); m.load_state_dict(ck["model"]); m.eval()
print("step", ck["step"])
# ---- weights
pc, pt = dict(m.ctx.named_parameters()), dict(m.tgt.named_parameters())
print(f"{'layer':<40} {'shape':>18} {'||Wc-Wt||/||Wc||':>17} {'cos(Wc,Wt)':>11}")
tot = [0, 0]
for k in pc:
    a, b = pc[k].detach().flatten(), pt[k].detach().flatten()
    tot[0] += ((a - b) ** 2).sum().item(); tot[1] += (a ** 2).sum().item()
    if a.numel() > 1000:
        print(f"{k:<40} {str(tuple(pc[k].shape)):>18} {((a-b).norm()/a.norm()).item():17.3f} {(a@b/(a.norm()*b.norm())).item():11.3f}")
print(f"all parameters: relative difference {np.sqrt(tot[0]/tot[1]):.3f}")
# hidden units can be permuted between nets, so weight comparison is only indicative; compare functions:
with torch.no_grad():
    Z = torch.cat([m.ctx(obs[k:k+500]) for k in range(0, len(obs), 500)]).numpy().astype(np.float64)
    Y = torch.cat([m.tgt(obs[k:k+500]) for k in range(0, len(obs), 500)]).numpy().astype(np.float64)
    A = [p for n, p in m.pred.named_parameters() if p.dim() == 2][0].detach().numpy().astype(np.float64)
Sf = S.reshape(-1); n = len(Z) // 2
Zc, Yc = Z - Z.mean(0), Y - Y.mean(0)
print(f"\nsame windows into both encoders (N={len(Z)})")
print(f"  ||Z - Y|| / ||Z|| (identity alignment)        {np.linalg.norm(Zc-Yc)/np.linalg.norm(Zc):.3f}")
Wz, Wy = ns["whiten_map"](Zc), ns["whiten_map"](Yc)
sv = np.linalg.svd((Zc @ Wz).T @ (Yc @ Wy) / len(Z), compute_uv=False)
print(f"  canonical correlations Z vs Y: top {np.round(sv[:4],3)}  median {np.median(sv):.3f}  min {sv.min():.3f}  #>0.9: {(sv>0.9).sum()}/124")
M, *_ = np.linalg.lstsq(Zc[:n], Yc[:n], rcond=None)
r2 = 1 - ((Yc[n:] - Zc[n:] @ M) ** 2).sum() / (Yc[n:] ** 2).sum()
U, _, Vt = np.linalg.svd(Zc[:n].T @ Yc[:n]); Rot = U @ Vt
r2o = 1 - ((Yc[n:] - Zc[n:] @ Rot) ** 2).sum() / (Yc[n:] ** 2).sum()
print(f"  held-out R² of Y from Z: best linear map {r2:.3f}   best pure rotation {r2o:.3f}")
print(f"  predictor A (Y_t+2 ≈ A Z_t): ||A - Rot^T||/||A|| {np.linalg.norm(A - Rot.T)/np.linalg.norm(A):.3f}   singular values {np.round(np.linalg.svd(A, compute_uv=False)[[0,10,40,80,123]],3)}")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
lp = lambda X: balanced_accuracy_score(Sf[n:], LogisticRegression(max_iter=3000, C=0.1).fit(X[:n], Sf[:n]).predict(X[n:]))
print(f"  linear probe for regime: context Z {lp(Zc):.3f}   target Y {lp(Yc):.3f}")
# regime subspaces of each, compared through the best rotation
mz = np.array([Zc[Sf == k].mean(0) for k in range(4)]); my = np.array([Yc[Sf == k].mean(0) for k in range(4)])
Qz = np.linalg.qr((mz - mz.mean(0)).T @ np.eye(4))[0][:, :3]; Qy = np.linalg.qr((my - my.mean(0)).T)[0][:, :3]
s = np.linalg.svd(np.linalg.qr(Rot.T @ Qz)[0].T @ Qy, compute_uv=False)
print(f"  angles between regime-mean subspaces (Z rotated into Y): {np.round(np.degrees(np.arccos(np.clip(s,-1,1))),1)}")
