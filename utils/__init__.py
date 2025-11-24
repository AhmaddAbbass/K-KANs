from .alignment import compute_alignment, centered_alignment
from .training import (
    train_kernel_alignment,
    kernel_ridge_train,
    kernel_ridge_predict,
)
from .datasets import make_two_gaussians, make_gp_regression_data
from .check_psd import check_psd

__all__ = [
    "compute_alignment",
    "centered_alignment",
    "train_kernel_alignment",
    "kernel_ridge_train",
    "kernel_ridge_predict",
    "make_two_gaussians",
    "make_gp_regression_data",
    "check_psd",
]
