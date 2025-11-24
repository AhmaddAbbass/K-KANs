import torch
from torch import nn

from .base import BaseKernel


class AdditiveRBFKernel(BaseKernel):
    """
    Additive kernel with 1D Gaussian (RBF) per coordinate.

    Math:
      x_i in R^p
      f_k(t) = w_k * t + b_k
      k_k(u, v) = exp( -0.5 * ((u - v) / ell_k)^2 )
      k(x_i, x_j) = sum_{k=1}^p k_k( f_k(x_i^k), f_k(x_j^k) )

    Implementation is fully vectorized over (i,j):
      - NO Python loops over samples
      - we only use broadcasting over dimensions.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

        # parameters of f_k(t) = w_k * t + b_k  (one pair per coordinate)
        self.weight_f = nn.Parameter(torch.randn(input_dim))
        self.bias_f = nn.Parameter(torch.zeros(input_dim))

        # log lengthscale per coordinate (ell_k = exp(log_ell_k) > 0)
        self.log_lengthscale = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (n, p)
        returns K: (n, n)
        """
        n, p = x.shape
        assert p == self.input_dim, "input_dim mismatch"

        # ---- 1) warp coordinates: f_x[i,k] = w_k * x[i,k] + b_k ----
        # weight_f, bias_f have shape (p,)
        # we reshape to (1, p) so they broadcast over rows (n, p)
        f_x = (
            x * self.weight_f.view(1, -1)
            + self.bias_f.view(1, -1)
        )  # shape (n, p)

        # ---- 2) pairwise differences per dimension ----
        # f_x.unsqueeze(1): (n, 1, p)
        # f_x.unsqueeze(0): (1, n, p)
        # broadcasted diff[i,j,k] = f_x[i,k] - f_x[j,k]
        diff = f_x.unsqueeze(1) - f_x.unsqueeze(0)  # (n, n, p)

        # ---- 3) lengthscales, broadcasted over (i,j) ----
        ell = torch.exp(self.log_lengthscale).view(1, 1, -1) + 1e-6  # (1,1,p)

        # ---- 4) 1D RBF per dimension, then sum over k ----
        K_per_dim = torch.exp(-0.5 * (diff / ell) ** 2)  # (n, n, p)
        K = K_per_dim.sum(dim=2)  # (n, n)

        return K
