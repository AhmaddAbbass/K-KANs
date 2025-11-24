import torch


def make_two_gaussians(
    n_per_class=100,
    dim=2,
    gap=2.0,
    device="cpu",
    seed=None,
):
    """
    Simple toy binary dataset.

    Construction:
      - Class +1 ~ N(+mu, I)
      - Class -1 ~ N(-mu, I)
      - mu = (gap, 0, ..., 0)

    So gap controls how separated the two clouds are along the first axis.
    """
    if seed is not None:
        torch.manual_seed(seed)

    mu = torch.zeros(dim, device=device)
    mu[0] = gap

    x_pos = torch.randn(n_per_class, dim, device=device) + mu
    x_neg = torch.randn(n_per_class, dim, device=device) - mu

    X = torch.cat([x_pos, x_neg], dim=0)

    y_pos = torch.ones(n_per_class, device=device)
    y_neg = -torch.ones(n_per_class, device=device)
    y = torch.cat([y_pos, y_neg], dim=0)

    return X, y


def make_gp_regression_data(
    n,
    p,
    kernel_fn,
    noise_std=0.1,
    device="cpu",
    seed=None,
):
    """
    GP-style synthetic regression data (Scenario A type).

    Steps:
      1) Sample X ~ N(0, I_p)  (n points in R^p)
      2) Build "true" Gram matrix K_true = kernel_fn(X)
      3) Sample f ~ N(0, K_true) via Cholesky
      4) Add Gaussian noise: y_i = f(x_i) + eps_i

    Returns:
      - X: (n, p)
      - y: (n,)
      - K_true: (n, n)  (the kernel we used to generate data)
    """
    if seed is not None:
        torch.manual_seed(seed)

    X = torch.randn(n, p, device=device)

    K_true = kernel_fn(X)

    jitter = 1e-5 * torch.eye(n, device=device, dtype=K_true.dtype)
    L = torch.linalg.cholesky(K_true + jitter)

    z = torch.randn(n, 1, device=device, dtype=K_true.dtype)
    f = (L @ z).view(-1)

    y = f + noise_std * torch.randn_like(f)

    return X, y, K_true
