"""Self-supervised networks (torch modules).  Wrapped as feature learners in cpbench/features/ssl.py.
Each network has one interface:
    loss(xc, xt) -> (scalar loss, dict of logged floats)     xc, xt: context / target windows (B, W, D)
    encode(x)    -> (B, d) window representation used by every readout
Methods that ignore the target window (AE, VAE, TS2Vec) simply do not use xt."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import Encoder
from .sigreg import SIGReg


class _Head(nn.Module):
    """Same encoder as `base` up to its last layer; own final linear layer."""
    def __init__(self, base, d):
        super().__init__()
        self.base = [base]                                               # not registered twice as a submodule
        self.last = nn.Linear(base.proj[-1].in_features, d)

    def forward(self, x):
        b = self.base[0]
        return self.last(b.proj[:-1](b.phi(x)))


class JEPA(nn.Module):
    """Linear- or MLP-predictor LeJEPA: MSE(y, A z) + lam * SIGReg, no EMA / stop-gradient (LeJEPA / LeWM).
    shared:        one encoder for context and target, one SIGReg on the pooled outputs.
    separate:      one shared backbone (feature dictionary), different final linear layers for context and target
                   (the theory's separate optimum: SVD of the cross-covariance in a common dictionary); mean of the
                   two SIGRegs.
    separate_full: two fully independent encoders (earlier experiments)."""
    def __init__(self, D, W, d=124, H=256, backbone="tcn", design="separate", predictor="linear", lam=0.5, **_):
        super().__init__()
        self.design, self.lam = design, lam
        self.ctx = Encoder(D, d, H, backbone)
        if design == "shared":
            self.tgt = self.ctx
        elif design == "separate":
            self.tgt = _Head(self.ctx, d)
        elif design == "separate_full":
            self.tgt = Encoder(D, d, H, backbone)
        else:
            raise ValueError(design)
        self.pred = nn.Linear(d, d, bias=False) if predictor == "linear" else nn.Sequential(
            nn.Linear(d, 512), nn.GELU(), nn.Linear(512, d))
        self.sigreg = SIGReg()

    def loss(self, xc, xt):
        if self.design == "shared":
            out = self.ctx(torch.cat([xc, xt])); z, y = out.chunk(2); reg = self.sigreg(out[None])
        else:
            z, y = self.ctx(xc), self.tgt(xt); reg = 0.5 * (self.sigreg(z[None]) + self.sigreg(y[None]))
        pred = (y - self.pred(z)).square().mean()
        return pred + self.lam * reg, dict(pred=pred.item(), reg=reg.item())

    def encode(self, x):
        return self.ctx(x)

    def encode_target(self, x):
        return self.tgt(x)


class MixtureJEPA(nn.Module):
    """LeJEPA with a switching linear predictor.  A state encoder reads only the last `state_window` samples (the time
    step t itself by default): z_t = S(x_t).  A gating encoder reads the whole context window ending at t and gives
    mixing weights w(t) = softmax(G(window) / temperature) over `experts` independent matrices A_k (random init):
        z_hat_{t+Delta} = sum_k w_k(t) A_k z_t,    target  y = S(x_{t+Delta})  (first state_window samples of the target)
    Loss: MSE(y, z_hat) + lam SIGReg(states) + balance (log experts - H(mean w)) (load balancing: without it the gate
    tends to send everything to one expert) + sharpness E[H(w)] (optional push towards one expert per window).
    Representation: the mixing weights w (clustering happens in that space); views: the state z and the gate logits."""
    def __init__(self, D, W, d=4, H=256, backbone="tcn", experts=8, state_window=1, temperature=1.0, lam=0.5,
                 balance=0.1, sharpness=0.05, **_):
        super().__init__()
        self.s, self.tau, self.lam, self.balance, self.sharpness = state_window, temperature, lam, balance, sharpness
        self.state = nn.Sequential(nn.Linear(state_window * D, 2 * H), nn.GELU(), nn.Linear(2 * H, 2 * H), nn.GELU(),
                                   nn.Linear(2 * H, d))
        self.gate = Encoder(D, experts, H, backbone)
        self.A = nn.Parameter(torch.randn(experts, d, d) / d ** 0.5)
        self.sigreg = SIGReg()

    def encode_state(self, x):
        return self.state(x[:, -self.s:].reshape(len(x), -1))

    def weights(self, x):
        return torch.softmax(self.gate(x) / self.tau, -1)

    def loss(self, xc, xt):
        z = self.encode_state(xc)
        y = self.state(xt[:, :self.s].reshape(len(xt), -1))
        w = self.weights(xc)                                             # (B, experts)
        pred_all = torch.einsum("kij,bj->bki", self.A, z)                # (B, experts, d)
        y_hat = (w[..., None] * pred_all).sum(1)
        pred = (y - y_hat).square().mean()
        reg = self.sigreg(torch.cat([z, y])[None])
        ent = lambda p: -(p * (p + 1e-9).log()).sum(-1)
        usage = ent(w.mean(0)); conf = ent(w).mean()
        loss = pred + self.lam * reg + self.balance * (torch.log(torch.tensor(float(w.shape[1]))) - usage) + self.sharpness * conf
        return loss, dict(pred=pred.item(), reg=reg.item(), usage_entropy=usage.item(), window_entropy=conf.item())

    def encode(self, x):
        return self.weights(x)


class Decoder(nn.Module):
    def __init__(self, d, W, D, H=256):
        super().__init__()
        self.W, self.D = W, D
        self.net = nn.Sequential(nn.Linear(d, 2 * H), nn.GELU(), nn.Linear(2 * H, 2 * H), nn.GELU(), nn.Linear(2 * H, W * D))

    def forward(self, z):
        return self.net(z).view(-1, self.W, self.D)

class AE(nn.Module):
    def __init__(self, D, W, d=124, H=256, backbone="tcn", **_):
        super().__init__()
        self.enc = Encoder(D, d, H, backbone); self.dec = Decoder(d, W, D, H)

    def loss(self, xc, xt):
        rec = (self.dec(self.enc(xc)) - xc).square().mean()
        return rec, dict(rec=rec.item())

    def encode(self, x):
        return self.enc(x)


class VAE(nn.Module):
    """beta-VAE with a diagonal Gaussian posterior; the representation is the posterior mean."""
    def __init__(self, D, W, d=124, H=256, backbone="tcn", beta=1.0, **_):
        super().__init__()
        self.beta = beta
        self.enc = Encoder(D, d, H, backbone, out_mult=2); self.dec = Decoder(d, W, D, H)

    def loss(self, xc, xt):
        mu, logvar = self.enc(xc).chunk(2, -1)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        rec = (self.dec(z) - xc).square().sum((1, 2)).mean()
        kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1).mean()
        return (rec + self.beta * kl) / (xc.shape[1] * xc.shape[2]), dict(rec=rec.item(), kl=kl.item())

    def encode(self, x):
        return self.enc(x).chunk(2, -1)[0]


class CPC(nn.Module):
    """InfoNCE between a linear prediction of the context embedding and the future (gapped) window embedding;
    in-batch negatives, cosine similarity with temperature."""
    def __init__(self, D, W, d=124, H=256, backbone="tcn", temperature=0.1, **_):
        super().__init__()
        self.enc = Encoder(D, d, H, backbone); self.pred = nn.Linear(d, d, bias=False); self.tau = temperature

    def loss(self, xc, xt):
        z, y = self.enc(xc), self.enc(xt)
        logits = F.normalize(self.pred(z), dim=-1) @ F.normalize(y, dim=-1).T / self.tau
        lab = torch.arange(len(z), device=z.device)
        l = F.cross_entropy(logits, lab)
        return l, dict(nce=l.item(), acc=(logits.argmax(1) == lab).float().mean().item())

    def encode(self, x):
        return self.enc(x)


class TS2VecLite(nn.Module):
    """Simplified TS2Vec (kept as an ablation): two augmented views (random input masking + jitter) of the SAME window
    on the shared backbone; hierarchical (max-pool by 2) instance-wise and temporal contrastive losses on per-timestep
    features.  No overlapping crops.  The window representation is the max-pool over time of the timestep features."""
    def __init__(self, D, W, d=124, H=256, backbone="tcn", mask_p=0.5, jitter=0.1, **_):
        super().__init__()
        self.enc = Encoder(D, d, H, backbone)          # only .phi (sequence features) is used
        self.head = nn.Linear(self.enc.phi.H, d); self.mask_p, self.jitter = mask_p, jitter

    def _seq(self, x):
        return self.head(self.enc.phi.sequence(x))                       # (B, W, d)

    def _aug(self, x):
        keep = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > self.mask_p).float()
        return x * keep + self.jitter * torch.randn_like(x)

    @staticmethod
    def _pair_loss(a, b):
        B, T, _ = a.shape
        loss = 0.0
        if B > 1:                                                         # instance contrast at each timestep
            z = torch.cat([a, b], 0).transpose(0, 1)                     # (T, 2B, d)
            sim = z @ z.transpose(1, 2); sim = sim - 1e9 * torch.eye(2 * B, device=a.device)
            lab = torch.cat([torch.arange(B, 2 * B), torch.arange(B)]).to(a.device)
            loss = loss + F.cross_entropy(sim.reshape(-1, 2 * B), lab.repeat(T))
        if T > 1:                                                         # temporal contrast within each instance
            z = torch.cat([a, b], 1)                                     # (B, 2T, d)
            sim = z @ z.transpose(1, 2); sim = sim - 1e9 * torch.eye(2 * T, device=a.device)
            lab = torch.cat([torch.arange(T, 2 * T), torch.arange(T)]).to(a.device)
            loss = loss + F.cross_entropy(sim.reshape(-1, 2 * T), lab.repeat(B))
        return loss

    def loss(self, xc, xt):
        a, b = self._seq(self._aug(xc)), self._seq(self._aug(xc))
        total, n = 0.0, 0
        while True:
            total = total + self._pair_loss(a, b); n += 1
            if a.shape[1] == 1: break
            a = F.max_pool1d(a.transpose(1, 2), 2).transpose(1, 2); b = F.max_pool1d(b.transpose(1, 2), 2).transpose(1, 2)
        total = total / n
        return total, dict(contrast=total.item())

    def encode(self, x):
        return self._seq(x).amax(1)


class _SamePadConv(nn.Module):
    def __init__(self, cin, cout, k, dilation):
        super().__init__()
        rf = (k - 1) * dilation + 1
        self.conv = nn.Conv1d(cin, cout, k, padding=rf // 2, dilation=dilation)
        self.trim = 1 if rf % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.trim] if self.trim else out


class _ConvBlock(nn.Module):
    def __init__(self, cin, cout, k, dilation, final=False):
        super().__init__()
        self.conv1, self.conv2 = _SamePadConv(cin, cout, k, dilation), _SamePadConv(cout, cout, k, dilation)
        self.proj = nn.Conv1d(cin, cout, 1) if cin != cout or final else None

    def forward(self, x):
        res = x if self.proj is None else self.proj(x)
        return self.conv2(F.gelu(self.conv1(F.gelu(x)))) + res


def _instance_contrast(z1, z2):
    B = z1.shape[0]
    if B == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], 0).transpose(0, 1)                           # (T, 2B, C)
    sim = z @ z.transpose(1, 2)
    logits = torch.tril(sim, -1)[:, :, :-1] + torch.triu(sim, 1)[:, :, 1:]   # drop the diagonal
    logits = -F.log_softmax(logits, -1)
    i = torch.arange(B, device=z1.device)
    return (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2


def _temporal_contrast(z1, z2):
    T = z1.shape[1]
    if T == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], 1)                                           # (B, 2T, C)
    sim = z @ z.transpose(1, 2)
    logits = torch.tril(sim, -1)[:, :, :-1] + torch.triu(sim, 1)[:, :, 1:]
    logits = -F.log_softmax(logits, -1)
    t = torch.arange(T, device=z1.device)
    return (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2


def hierarchical_contrast(z1, z2, alpha=0.5, temporal_unit=0):
    """TS2Vec's loss: instance-wise (weight alpha) and temporal (1 - alpha) contrast at every max-pool-by-2 scale."""
    loss, d = 0.0, 0
    while z1.shape[1] > 1:
        loss = loss + alpha * _instance_contrast(z1, z2)
        if d >= temporal_unit:
            loss = loss + (1 - alpha) * _temporal_contrast(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), 2).transpose(1, 2); z2 = F.max_pool1d(z2.transpose(1, 2), 2).transpose(1, 2)
    if z1.shape[1] == 1:
        loss = loss + alpha * _instance_contrast(z1, z2); d += 1
    return loss / d


class TS2Vec(nn.Module):
    """TS2Vec (Yue et al., AAAI 2022), following the reference implementation: input linear layer, binomial timestamp
    masking of the latent (training only), dilated residual CNN (depth blocks, dilation 2^i, hidden width `hidden`,
    output d), two random overlapping crops of each training segment, hierarchical instance + temporal contrast on the
    overlap.  Positions outside a crop are zero-filled and masked, as the reference pads with NaN.  The window
    representation is the max-pool over time (encoding_window="full_series").  The training segment is the context
    window (the target window is unused).  Reference defaults: hidden 64, depth 10, output 320 (here d, by default the
    benchmark's common embedding size)."""
    def __init__(self, D, W, d=124, hidden=64, depth=10, mask_p=0.5, alpha=0.5, temporal_unit=0, **_):
        super().__init__()
        self.mask_p, self.alpha, self.temporal_unit = mask_p, alpha, temporal_unit
        self.inp = nn.Linear(D, hidden)
        chans = [hidden] * depth + [d]
        self.blocks = nn.Sequential(*[_ConvBlock(chans[i - 1] if i else hidden, chans[i], 3, 2 ** i, final=i == len(chans) - 1)
                                      for i in range(len(chans))])
        self.drop = nn.Dropout(0.1)

    def _seq(self, x, valid=None):
        h = self.inp(x)                                                  # (B, T, hidden)
        if self.training:
            keep = torch.rand(h.shape[:2], device=x.device) > self.mask_p
            valid = keep if valid is None else keep & valid
        if valid is not None:
            h = h * valid[..., None]
        return self.drop(self.blocks(h.transpose(1, 2))).transpose(1, 2)  # (B, T, d)

    @staticmethod
    def _take(x, start, length):
        """Per-sample crops x[b, start_b : start_b + length], zero outside [0, T); returns crops and validity mask."""
        B, T, _ = x.shape
        idx = start[:, None] + torch.arange(length, device=x.device)[None]
        valid = (idx >= 0) & (idx < T)
        out = torch.gather(x, 1, idx.clamp(0, T - 1)[..., None].expand(-1, -1, x.shape[2]))
        return out * valid[..., None], valid

    def loss(self, xc, xt):
        B, T, _ = xc.shape
        rnd = lambda lo, hi: int(torch.randint(lo, hi, (1,)))
        crop_l = rnd(2 ** (self.temporal_unit + 1), T + 1)
        left = rnd(0, T - crop_l + 1); right = left + crop_l
        eleft = rnd(0, left + 1); eright = rnd(right, T + 1)
        offset = torch.randint(-eleft, T - eright + 1, (B,), device=xc.device)
        a, va = self._take(xc, offset + eleft, right - eleft)
        b, vb = self._take(xc, offset + left, eright - left)
        z1 = self._seq(a, va)[:, -crop_l:]
        z2 = self._seq(b, vb)[:, :crop_l]
        l = hierarchical_contrast(z1, z2, self.alpha, self.temporal_unit)
        return l, dict(contrast=l.item(), crop=float(crop_l))

    def encode(self, x):
        return self._seq(x).amax(1)


def latent_prediction(pred, z, y):
    """MSE(y, pred(z)) divided by the variance of y: scale-free, so the encoder cannot shrink its latents (and the
    decoder rescale them) to make the prediction trivially good."""
    return (y - pred(z)).square().mean() / (y.var(0).mean() + 1e-6)


class PredAE(nn.Module):
    """Autoencoder of both windows plus a linear predictor in latent space: rec(xc) + rec(xt) + mu * pred, with pred the
    normalised MSE between enc(xt) and A enc(xc) (A linear, as in the linear-predictor JEPA)."""
    def __init__(self, D, W, d=124, H=256, backbone="tcn", mu=1.0, **_):
        super().__init__()
        self.mu = mu
        self.enc = Encoder(D, d, H, backbone); self.dec = Decoder(d, W, D, H)
        self.pred = nn.Linear(d, d, bias=False)

    def loss(self, xc, xt):
        z, y = self.enc(torch.cat([xc, xt])).chunk(2)
        rec = 0.5 * ((self.dec(z) - xc).square().mean() + (self.dec(y) - xt).square().mean())
        pred = latent_prediction(self.pred, z, y)
        return rec + self.mu * pred, dict(rec=rec.item(), pred=pred.item())

    def encode(self, x):
        return self.enc(x)


class PredVAE(nn.Module):
    """beta-VAE of both windows plus a linear predictor in latent space: the target posterior mean is predicted from a
    posterior sample of the context, A z_c (normalised MSE, weight mu).  Representation: the posterior mean."""
    def __init__(self, D, W, d=124, H=256, backbone="tcn", beta=1.0, mu=1.0, **_):
        super().__init__()
        self.beta, self.mu = beta, mu
        self.enc = Encoder(D, d, H, backbone, out_mult=2); self.dec = Decoder(d, W, D, H)
        self.pred = nn.Linear(d, d, bias=False)

    def loss(self, xc, xt):
        x = torch.cat([xc, xt])
        m, logvar = self.enc(x).chunk(2, -1)
        z = m + torch.randn_like(m) * (0.5 * logvar).exp()
        rec = (self.dec(z) - x).square().sum((1, 2)).mean()
        kl = 0.5 * (m.square() + logvar.exp() - 1 - logvar).sum(-1).mean()
        zc, _ = z.chunk(2); _, mt = m.chunk(2)
        pred = latent_prediction(self.pred, zc, mt)
        return (rec + self.beta * kl) / (xc.shape[1] * xc.shape[2]) + self.mu * pred, dict(rec=rec.item(), kl=kl.item(), pred=pred.item())

    def encode(self, x):
        return self.enc(x).chunk(2, -1)[0]
