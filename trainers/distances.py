import ot
import torch
from typing import Literal
import torch.backends.cuda as cuda_backends


def mse_potential(z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    return (z0 - z1).pow(2).sum(dim=1)


def wasserstein_distance(
    z0: torch.Tensor,
    z1: torch.Tensor,
    eps: float = 0.0
) -> torch.Tensor:
    a = torch.ones(z0.shape[0], device=z0.device) / z0.shape[0]
    b = torch.ones(z1.shape[0], device=z1.device) / z1.shape[0]
    C = torch.cdist(z0, z1, p=2) ** 2

    with torch.no_grad():
        if eps > 0:
            pi: torch.Tensor = ot.sinkhorn(  # type: ignore
                a, b, C / C.max(), reg=eps, numItermax=100)
        else:
            pi: torch.Tensor = ot.emd(a, b, C)  # type: ignore

    return torch.sum(pi * C)


def mmd_distance(
    z0: torch.Tensor,
    z1: torch.Tensor,
    kernel: Literal["rbf", "linear"] = "rbf",
    bandwidth: float = 1.0
) -> torch.Tensor:
    if kernel == "rbf":
        def rbf_kernel(x, y, bandwidth):
            dist = torch.cdist(x, y, p=2) ** 2
            return torch.exp(-dist / (2 * bandwidth ** 2))
        
        K_xx = rbf_kernel(z0, z0, bandwidth)
        K_yy = rbf_kernel(z1, z1, bandwidth)
        K_xy = rbf_kernel(z0, z1, bandwidth)
    elif kernel == "linear":
        K_xx = z0 @ z0.T
        K_yy = z1 @ z1.T
        K_xy = z0 @ z1.T
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd


def procrustes_distance(data1: torch.Tensor,
                     data2: torch.Tensor,
                     eps: float = 1e-12):
    """
    Torch implementation of SciPy's scipy.spatial.procrustes.

    Parameters
    ----------
    data1 : (n, d) torch.Tensor
        Reference point set. Rows are points, columns are dimensions.
    data2 : (n, d) torch.Tensor
        Target point set to be transformed toward data1.
    eps : float
        Small constant to avoid division by zero for degenerate inputs.

    Returns
    -------
    mtx1 : (n, d) torch.Tensor
        data1 centered and scaled to unit Frobenius norm.
    mtx2 : (n, d) torch.Tensor
        data2 centered, scaled, and optimally rotated/reflected and dilated to align with mtx1.
    disparity : torch.Tensor (scalar)
        Sum of squared differences ||mtx1 - mtx2||_F^2 (M^2 in SciPy docs).

    Notes
    -----
    - Mirrors the behavior described in SciPy's procrustes docs: center, scale each set to unit
      Frobenius norm, then find optimal rotation/reflection and a single global scale to minimize
      squared differences【2-0】【2-2】.
    - Autograd-friendly and works on CPU or GPU, depending on tensor device.
    """

    def _safe_svd(A):
        # Ensure supported dtype for SVD on CUDA: float32/float64/complex types
        if A.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
            A = A.to(torch.float32)

        if A.is_cuda:
            try:
                # Default backend choice
                return torch.linalg.svd(A, full_matrices=False)
            except RuntimeError as e:
                # 1) Prefer MAGMA over cuSOLVER and retry
                try:
                    cuda_backends.preferred_linalg_library('magma')  # switch backend【8-2】
                    return torch.linalg.svd(A, full_matrices=False)
                except Exception:
                    pass
                # 2) Try alternate cuSOLVER drivers
                for drv in ('gesvdj', 'gesvd', 'gesvda'):  # choose algorithm【8-0】
                    try:
                        return torch.linalg.svd(A, full_matrices=False, driver=drv)
                    except Exception:
                        continue
                # 3) Final fallback: do SVD on CPU and move results back
                U, S, Vh = torch.linalg.svd(A.cpu(), full_matrices=False)
                return U.to(A.device), S.to(A.device), Vh.to(A.device)
        else:
            return torch.linalg.svd(A, full_matrices=False)


    if data1.ndim != 2 or data2.ndim != 2:
        raise ValueError("Inputs must be 2D tensors of shape (n, d).")
    if data1.shape != data2.shape:
        raise ValueError(f"Shape mismatch: {data1.shape} vs {data2.shape}.")

    dtype = data1.dtype
    if data2.dtype != dtype:
        data2 = data2.to(dtype)

    # 1) Center each dataset at the origin
    data1_centered = data1 - data1.mean(dim=0, keepdim=True)
    data2_centered = data2 - data2.mean(dim=0, keepdim=True)

    # 2) Scale each to unit Frobenius norm (avoid divide by zero)
    norm1 = torch.linalg.norm(data1_centered)
    norm2 = torch.linalg.norm(data2_centered)
    norm1 = torch.clamp(norm1, min=eps)
    norm2 = torch.clamp(norm2, min=eps)
    mtx1 = data1_centered / norm1
    mtx2 = data2_centered / norm2

    # 3) Optimal orthogonal alignment via SVD
    #    A = X^T Y, A = U Σ V^T -> R = U V^T
    A = mtx1.transpose(0, 1) @ mtx2
    with torch.no_grad():
        U, S, Vh = _safe_svd(A)
    R = U @ Vh

    # 4) Optimal global scaling (since both are unit-norm, s = sum(S))
    scale = S.sum()

    # 5) Apply transform to mtx2
    mtx2 = (mtx2 @ R) * scale

    # 6) Disparity: sum of squared differences
    disparity = torch.sum((mtx1 - mtx2) ** 2)

    return disparity
