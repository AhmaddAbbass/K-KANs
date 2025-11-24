import torch
from torch import nn

from .base import BaseKernel


class MixedAdditiveKernel(BaseKernel):
    """
    Additive kernel with a *mixture* of 1D primitives per coordinate.

    For coordinate k:

      f_k(t) = w_k * t + b_k
      type = k mod 3, so we cycle through:

        0 -> RBF:
             k_k(u, v) = exp( -0.5 * ((u - v) / ell_k)^2 )

        1 -> Laplacian:
             k_k(u, v) = exp( -|u - v| / ell_k )

        2 -> Cosine:
             k_k(u, v) = (u v) / (|u| |v|)   (per-dimension cosine)

    Final kernel:
      k(x_i, x_j) = sum_k k_k( f_k(x_i^k), f_k(x_j^k) )

    Idea: different coordinates can emphasize different “notions of similarity”.
    Implementation:
      - fully vectorized over (i,j)
      - we group dimensions by type (0,1,2) so we never loop over samples.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

        # same warp structure as additive RBF
        self.weight_f = nn.Parameter(torch.randn(input_dim))
        self.bias_f = nn.Parameter(torch.zeros(input_dim))
        self.log_lengthscale = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (n, p)
        returns K: (n, n)
        """
        n, p = x.shape
        assert p == self.input_dim, "input_dim mismatch"

        # ---- 1) warp coordinates: f_x[i,k] = w_k * x[i,k] + b_k ----
        f_x = (
            x * self.weight_f.view(1, -1)
            + self.bias_f.view(1, -1)
        )  # (n, p)

        # one lengthscale per coordinate (for RBF/Laplacian parts)
        ell = torch.exp(self.log_lengthscale) + 1e-6  # (p,)

        # we'll accumulate all pieces here
        K = torch.zeros(n, n, device=x.device, dtype=x.dtype)

        # decide which dimension goes to which primitive
        all_idx = torch.arange(p, device=x.device)
        idx_rbf = all_idx[all_idx % 3 == 0]
        idx_lap = all_idx[all_idx % 3 == 1]
        idx_cos = all_idx[all_idx % 3 == 2]

        # helper to build pairwise differences for a subset of dims
        def pairwise_diffs(sub_f_x):
            # sub_f_x: (n, d)
            # outcome diff[i,j,d] = sub_f_x[i,d] - sub_f_x[j,d]
            return sub_f_x.unsqueeze(1) - sub_f_x.unsqueeze(0)

        # --- RBF block ------------------------------------------------------
        if idx_rbf.numel() > 0:
            f_rbf = f_x[:, idx_rbf]  # (n, d_r)
            ell_rbf = ell[idx_rbf].view(1, 1, -1)  # (1,1,d_r)

            diff_rbf = pairwise_diffs(f_rbf)  # (n,n,d_r)
            K_rbf = torch.exp(-0.5 * (diff_rbf / ell_rbf) ** 2).sum(dim=2)
            K = K + K_rbf

        # --- Laplacian block -----------------------------------------------
        if idx_lap.numel() > 0:
            f_lap = f_x[:, idx_lap]  # (n, d_l)
            ell_lap = ell[idx_lap].view(1, 1, -1)

            diff_lap = pairwise_diffs(f_lap)
            K_lap = torch.exp(-torch.abs(diff_lap) / ell_lap).sum(dim=2)
            K = K + K_lap

        # --- Cosine block (scalar cosine per dimension) --------------------
        if idx_cos.numel() > 0:
            f_cos = f_x[:, idx_cos]  # (n, d_c)

            # f_norm[i,k] = |f_cos[i,k]|, just scalar norms per dim
            f_norm = torch.sqrt(f_cos ** 2 + 1e-8)

            # shapes:
            #   u: (n,1,d_c), v: (1,n,d_c)
            #   nu: (n,1,d_c), nv: (1,n,d_c)
            u = f_cos.unsqueeze(1)
            v = f_cos.unsqueeze(0)
            nu = f_norm.unsqueeze(1)
            nv = f_norm.unsqueeze(0)

            cos = (u * v) / (nu * nv + 1e-8)  # (n,n,d_c)
            K_cos = cos.sum(dim=2)
            K = K + K_cos

        return K
