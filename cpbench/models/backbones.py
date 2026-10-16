"""Window encoders.  Every backbone maps windows (B, W, D) to per-timestep features (B, W, H) via `sequence` and
to a window vector via `forward`; Encoder adds the projection head to the embedding dimension d."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TCN(nn.Module):
    """Dilated residual TCN (long_run.BigTCN), mean+max pooled."""
    def __init__(self, D, H=256, levels=5, k=3):
        super().__init__()
        self.inp = nn.Conv1d(D, H, 1)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv1d(H, H, k, dilation=2 ** i, padding=2 ** i), nn.GroupNorm(8, H), nn.GELU(),
            nn.Conv1d(H, H, k, dilation=2 ** i, padding=2 ** i)) for i in range(levels)])
        self.out_dim, self.H = 2 * H, H

    def sequence(self, x):
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = h + b(F.gelu(h))
        return F.gelu(h).transpose(1, 2)                                 # (B, W, H)

    def forward(self, x):
        h = self.sequence(x)
        return torch.cat([h.mean(1), h.amax(1)], -1)


class GRUEnc(nn.Module):
    def __init__(self, D, H=256, layers=2):
        super().__init__()
        self.rnn = nn.GRU(D, H, layers, batch_first=True)
        self.out_dim, self.H = 2 * H, H

    def sequence(self, x):
        return self.rnn(x)[0]

    def forward(self, x):
        h = self.sequence(x)
        return torch.cat([h[:, -1], h.mean(1)], -1)


class TransformerEnc(nn.Module):
    def __init__(self, D, H=256, layers=4, heads=8, max_len=512):
        super().__init__()
        self.inp = nn.Linear(D, H)
        self.pos = nn.Parameter(torch.randn(1, max_len, H) * 0.02)
        self.tr = nn.TransformerEncoder(nn.TransformerEncoderLayer(H, heads, 4 * H, batch_first=True, norm_first=True,
                                                                   activation="gelu", dropout=0.0), layers)
        self.out_dim, self.H = 2 * H, H

    def sequence(self, x):
        return self.tr(self.inp(x) + self.pos[:, : x.shape[1]])

    def forward(self, x):
        h = self.sequence(x)
        return torch.cat([h.mean(1), h.amax(1)], -1)


BACKBONES = dict(tcn=TCN, gru=GRUEnc, transformer=TransformerEnc)


class Encoder(nn.Module):
    """backbone -> MLP head -> d (long_run.BigEncoder)."""
    def __init__(self, D, d=124, H=256, backbone="tcn", out_mult=1):
        super().__init__()
        self.phi = BACKBONES[backbone](D, H)
        self.proj = nn.Sequential(nn.Linear(self.phi.out_dim, 2 * H), nn.BatchNorm1d(2 * H), nn.GELU(),
                                  nn.Linear(2 * H, d * out_mult))

    def forward(self, x):
        return self.proj(self.phi(x))
