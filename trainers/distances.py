import ot
import torch
from typing import Literal


def mse_potential(z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    dists = torch.square(torch.norm(z0 - z1, p=2, dim=1))  # type: ignore
    return torch.reciprocal(dists).mean()


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
