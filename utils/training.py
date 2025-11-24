import torch
import torch.optim as optim

from .alignment import compute_alignment


def train_kernel_alignment(
    kernel_model,
    X,
    y,
    lr=0.05,
    num_epochs=200,
    print_every=50,
):
    """
    Tiny training loop that only optimizes the kernel via alignment.

      A(theta) = (1/n^2) sum_ij K_theta(x_i,x_j) y_i y_j

    We do:
      loss(theta) = -A(theta)
      then a standard gradient step with Adam.

    This does *not* train a classifier head, just the kernel parameters.
    """

    optimizer = optim.Adam(kernel_model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()

        alignment, _ = compute_alignment(kernel_model, X, y)
        loss = -alignment

        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % print_every == 0:
            print(f"epoch {epoch:3d} | alignment = {alignment.item():.4f}")


def kernel_ridge_train(K, y, lam=1e-2):
    """
    Kernel ridge regression / classification.

    Closed form solution:
      alpha = (K + lam I)^(-1) y

    Where:
      - K is the Gram matrix on the training set
      - lam is the regularization strength
      - y are the targets (shape (n,))
    """
    n = K.shape[0]
    device = K.device
    dtype = K.dtype

    I = torch.eye(n, device=device, dtype=dtype)
    A = K + lam * I

    y_vec = y.view(-1, 1)  # (n,1)

    alpha = torch.linalg.solve(A, y_vec)  # (n,1)

    return alpha.view(-1)


def kernel_ridge_predict(K_test, alpha):
    """
    Given K_test[i,j] = K(x_test_i, x_train_j):

      f(x_test_i) = sum_j alpha_j K_test[i,j]

    We just do this for all test points at once as a matrix-vector product.
    """
    return K_test @ alpha
