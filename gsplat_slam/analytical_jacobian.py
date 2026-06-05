"""Analytical Jacobian computation for GaussianSplatFactor.

Computes the Jacobian of the photometric residual w.r.t. the Pose3 tangent
space (se(3)) using central differences through the se(3) generators.

GTSAM convention: right-exponential update, i.e. T_new = T * Exp(xi).
The error is left-invariant. We compute d(residual)/d(xi) at xi=0.

The viewmat is parameterized as: viewmat(xi) = Exp(-xi) * viewmat0
First-order: viewmat(xi) ≈ (I - hat(xi)) * viewmat0

Central differences give O(eps^2) accuracy (matching GTSAM's numerical_derivative).
Total cost: 12 GPU renders (2 per tangent direction), all on GPU with no
GTSAM Pose3.retract() calls in the inner loop.
"""

import torch
import numpy as np


_GENERATORS = None


def _get_generators(device):
    """Get cached se(3) generator matrices on the given device.

    GTSAM convention: xi = [omega_x, omega_y, omega_z, v_x, v_y, v_z]
    hat(xi) = | [omega]_x  v |
              |    0        0 |
    where [omega]_x is the skew-symmetric matrix:
    [omega]_x = |  0   -wz   wy |
                |  wz   0   -wx |
                | -wy   wx   0  |
    """
    global _GENERATORS
    if _GENERATORS is None or _GENERATORS.device != device:
        G = torch.zeros(6, 4, 4, device=device)
        # rot_x: [omega]_x for omega=[1,0,0]
        G[0, 1, 2] = -1.0; G[0, 2, 1] = 1.0
        # rot_y: [omega]_x for omega=[0,1,0]
        G[1, 0, 2] = 1.0; G[1, 2, 0] = -1.0
        # rot_z: [omega]_x for omega=[0,0,1]
        G[2, 0, 1] = -1.0; G[2, 1, 0] = 1.0
        # trans_x
        G[3, 0, 3] = 1.0
        # trans_y
        G[4, 1, 3] = 1.0
        # trans_z
        G[5, 2, 3] = 1.0
        _GENERATORS = G
    return _GENERATORS


def compute_analytical_jacobian(
    gaussian_map,
    target_image,
    K,
    pixel_indices,
    W, H,
    viewmat0,
    device="cuda",
    eps=1e-4,
):
    """Compute Jacobian of photometric residual w.r.t. se(3).

    Uses central differences along the 6 se(3) generator directions.
    Total cost: 12 renders (2 per direction) + 1 base render, all on GPU.
    Accuracy: O(eps^2), matching GTSAM's numerical_derivative.

    Args:
        gaussian_map: GaussianMap with means, quats, scales, opacities, colors
        target_image: (H, W, 3) tensor on device
        K: (3, 3) intrinsics tensor on device
        pixel_indices: (n_samples, 2) numpy array of (row, col)
        W, H: image dimensions
        viewmat0: (4, 4) tensor, the current camera-from-world transform
        device: torch device
        eps: finite difference step size

    Returns:
        residual: (n_residuals,) numpy array
        jacobian: (n_residuals, 6) numpy array
    """
    from gsplat_slam.renderer import render_gaussians, compute_photometric_residual

    generators = _get_generators(torch.device(device))
    vm0 = viewmat0.detach()

    means = gaussian_map.means.detach()
    quats = gaussian_map.quats.detach()
    scales = gaussian_map.scales.detach()
    opacities = gaussian_map.opacities.detach()
    colors = gaussian_map.colors.detach()
    K_d = K.detach()
    target_d = target_image.detach()

    with torch.no_grad():
        # Base render (for residual only)
        rendered_base, _, _ = render_gaussians(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmat=vm0, K=K_d, W=W, H=H,
        )
        residual_base = compute_photometric_residual(rendered_base, target_d, pixel_indices)
        residual_np = residual_base.cpu().numpy()

        # Central differences: 12 renders (+ and - for each of 6 directions)
        n = len(residual_np)
        jacobian = np.zeros((n, 6), dtype=np.float64)

        I4 = torch.eye(4, device=device)
        for i in range(6):
            vm_plus = (I4 - eps * generators[i]) @ vm0
            vm_minus = (I4 + eps * generators[i]) @ vm0

            rendered_plus, _, _ = render_gaussians(
                means=means, quats=quats, scales=scales,
                opacities=opacities, colors=colors,
                viewmat=vm_plus, K=K_d, W=W, H=H,
            )
            rendered_minus, _, _ = render_gaussians(
                means=means, quats=quats, scales=scales,
                opacities=opacities, colors=colors,
                viewmat=vm_minus, K=K_d, W=W, H=H,
            )

            res_plus = compute_photometric_residual(rendered_plus, target_d, pixel_indices)
            res_minus = compute_photometric_residual(rendered_minus, target_d, pixel_indices)
            jacobian[:, i] = (res_plus.cpu().numpy() - res_minus.cpu().numpy()) / (2 * eps)

    return residual_np, jacobian
