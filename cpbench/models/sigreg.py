import torch


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)

    Verbatim from lucas-maes/le-wm, module.py (the LeWM implementation of LeJEPA's SIGReg):
    Epps-Pulley test of every random 1-D projection against N(0,1), on 17 knots in [0,3].
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()  # average over projections and time


def batch_whiten(z, eps=1e-4):
    """Exact in-batch Cholesky whitening (W-MSE / VAMPnet style)."""
    zc = z - z.mean(0, keepdim=True)
    cov = zc.T @ zc / (z.shape[0] - 1) + eps * torch.eye(z.shape[1], device=z.device)
    L = torch.linalg.cholesky(cov.cpu()).to(z.device)                    # cholesky is not implemented on MPS
    return torch.linalg.solve_triangular(L, zc.T, upper=False).T
