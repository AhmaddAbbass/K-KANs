import torch

class FixedRBFKernel(torch.nn.Module):
    """
    Simple baseline kernel:
      k(x_i, x_j) = exp( - ||x_i - x_j||^2 / (2 * ell^2) )

    - No learning here, ell (lengthscale) is fixed.
    - This gives us a classic RBF to compare against our learned additive kernels.
    """

    def __init__(self, input_dim, lengthscale=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.lengthscale = float(lengthscale)

    def forward(self, x):
        """
        x: (n, p)
        returns K: (n, n)
        """
        n, p = x.shape
        assert p == self.input_dim, "input_dim mismatch in FixedRBFKernel"

        # pairwise differences: x_i - x_j
        diff = x.unsqueeze(1) - x.unsqueeze(0)   # (n, n, p)

        # squared L2 distance ||x_i - x_j||^2
        dist2 = (diff ** 2).sum(dim=2)           # (n, n)

        ell2 = self.lengthscale ** 2 + 1e-12

        K = torch.exp(-0.5 * dist2 / ell2)       # (n, n)

        return K
