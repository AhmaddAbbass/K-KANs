import torch


def check_psd(K, tol=1e-6, verbose=True):
    """
    Quick numerical check if K is positive semi-definite.

    Procedure:
      - symmetrize (just in case of tiny asymmetries from numerics)
      - compute eigenvalues
      - look at the minimum eigenvalue

    If min eigenvalue >= -tol we call it "PSD enough".
    """
    K_sym = 0.5 * (K + K.t())

    # eigvalsh -> symmetric eigenvalues, real
    eigvals = torch.linalg.eigvalsh(K_sym)
    min_eig = eigvals.min().item()
    is_psd = min_eig >= -tol

    if verbose:
        print(f"min eigenvalue = {min_eig:.4e} | PSD? {is_psd}")

    return is_psd, min_eig
