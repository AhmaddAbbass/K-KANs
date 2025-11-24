import torch


def compute_alignment(kernel_model, X, y):
    """
    Basic (uncentered) alignment between kernel and labels.

    A(theta) = (1 / n^2) sum_{i,j} K_theta(x_i, x_j) y_i y_j

    Vectorized story:
      - build Gram matrix K = K_theta(X)
      - build label Gram L = y y^T
      - alignment is just the average of K_ij * L_ij
    """
    K = kernel_model(X)        # (n, n)
    y = y.view(-1, 1)          # (n, 1)

    label_gram = y @ y.t()     # (n, n), entry (i,j) = y_i * y_j

    alignment = (K * label_gram).mean()  # scalar

    return alignment, K


def centered_alignment(K, T):
    """
    Centered, normalized alignment A_c(K, T).

    Given two Gram matrices K and T (n x n):

      H = I - (1/n) 11^T     (centering in feature space)
      Kc = H K H
      Tc = H T H

      A_c = <Kc, Tc>_F / (||Kc||_F * ||Tc||_F)

    This is the "centered alignment" definition from kernel alignment papers.
    """
    n = K.shape[0]
    device = K.device
    dtype = K.dtype

    # centering matrix H
    H = torch.eye(n, device=device, dtype=dtype) - \
        torch.ones(n, n, device=device, dtype=dtype) / n

    Kc = H @ K @ H
    Tc = H @ T @ H

    # Frobenius inner product in numerator
    num = (Kc * Tc).sum()

    # Frobenius norms in denominator
    denom1 = torch.sqrt((Kc * Kc).sum() + 1e-12)
    denom2 = torch.sqrt((Tc * Tc).sum() + 1e-12)

    return num / (denom1 * denom2 + 1e-12)

def compute_centered_alignment_loss(kernel_model, X, y):
    K = kernel_model(X)
    y = y.view(-1,1)
    T = y @ y.t()
    return -centered_alignment(K, T)

