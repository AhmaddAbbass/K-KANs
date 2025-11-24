import torch
import torch.nn as nn


class BaseKernel(nn.Module):
    """
    Base class for kernels.

    Convention:
      - forward(x) takes x of shape (n_samples, input_dim)
      - returns Gram matrix K of shape (n_samples, n_samples)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Child kernels must implement forward().")

    def gram(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
