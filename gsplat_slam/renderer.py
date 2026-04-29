"""Thin wrapper around gsplat's rasterization for use as a GTSAM factor."""

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from gsplat import rasterization
    HAS_GSPLAT = True
except ImportError:
    HAS_GSPLAT = False


def sample_pixel_indices(H: int, W: int, n_samples: int, rng=None) -> np.ndarray:
    """Uniformly sample pixel coordinates from an H x W image.

    Returns (n_samples, 2) array of (row, col) indices.
    """
    if rng is None:
        rng = np.random.default_rng()
    total = H * W
    n_samples = min(n_samples, total)
    flat = rng.choice(total, size=n_samples, replace=False)
    rows = flat // W
    cols = flat % W
    return np.stack([rows, cols], axis=1)


def render_gaussians(
    means: "torch.Tensor",       # (N, 3)
    quats: "torch.Tensor",       # (N, 4) wxyz
    scales: "torch.Tensor",      # (N, 3)
    opacities: "torch.Tensor",   # (N,)
    colors: "torch.Tensor",      # (N, 3) or (N, sh_degree, 3)
    viewmat: "torch.Tensor",     # (4, 4) world-to-camera
    K: "torch.Tensor",           # (3, 3) intrinsics
    W: int,
    H: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
):
    """Render Gaussians from a given camera pose.

    Args:
        means: Gaussian centers in world frame.
        quats: Quaternions (wxyz) for each Gaussian.
        scales: Log-scale of each Gaussian along 3 axes.
        opacities: Sigmoid-space opacity for each Gaussian.
        colors: Per-Gaussian RGB (or SH coefficients).
        viewmat: 4x4 world-to-camera transform.
        K: 3x3 camera intrinsics.
        W, H: Image dimensions.

    Returns:
        rendered: (H, W, 3) rendered image tensor.
        alpha: (H, W, 1) alpha/opacity map.
        meta: dict with additional rasterization info.
    """
    assert HAS_GSPLAT, "gsplat is required for rendering. pip install gsplat"

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    rendered, alpha, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities.sigmoid(),
        colors=colors,
        viewmats=viewmat[None],  # (1, 4, 4)
        Ks=torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                        device=means.device, dtype=means.dtype)[None],
        width=W,
        height=H,
        near_plane=near_plane,
        far_plane=far_plane,
        render_mode="RGB",
    )
    return rendered[0], alpha[0], meta


def compute_photometric_residual(
    rendered: "torch.Tensor",   # (H, W, 3)
    target: "torch.Tensor",     # (H, W, 3)
    pixel_indices: np.ndarray,  # (n_samples, 2) row, col
) -> "torch.Tensor":
    """Compute residual vector at sampled pixels.

    Returns (n_samples * 3,) tensor of per-channel differences.
    """
    rows = pixel_indices[:, 0]
    cols = pixel_indices[:, 1]
    rendered_samples = rendered[rows, cols]  # (n, 3)
    target_samples = target[rows, cols]      # (n, 3)
    return (rendered_samples - target_samples).reshape(-1)
