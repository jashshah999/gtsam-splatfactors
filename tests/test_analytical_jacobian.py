"""Verify analytical Jacobian matches numerical derivative.

Frank Dellaert's requirement: use numerical_derivative-style unit tests
to ensure the autograd-based Jacobian handles GTSAM's right-exponential
update convention correctly.
"""

import numpy as np
import pytest

try:
    import torch
    import gtsam
    from gsplat import rasterization
    HAS_ALL = torch.cuda.is_available()
except ImportError:
    HAS_ALL = False

requires_all = pytest.mark.skipif(not HAS_ALL, reason="Requires CUDA + gsplat + gtsam")


def _make_scene(device="cuda"):
    from gsplat_slam.map import GaussianMap
    n = 200
    means = np.zeros((n, 3), dtype=np.float32)
    means[:, 0] = np.random.uniform(-1.5, 1.5, n)
    means[:, 1] = np.random.uniform(-1.5, 1.5, n)
    means[:, 2] = np.random.uniform(3, 5, n)
    colors = np.random.rand(n, 3).astype(np.float32)
    gmap = GaussianMap.from_pointcloud(means, colors, device=device)
    gmap.scales.data.fill_(0.0)
    gmap.opacities.data.fill_(0.99)
    return gmap


@requires_all
def test_analytical_matches_numerical():
    """Core test: analytical Jacobian must match numerical within tolerance."""
    from gsplat_slam.factor import GaussianSplatFactor
    from gsplat_slam.renderer import render_gaussians, sample_pixel_indices
    from gsplat_slam.pose_utils import matrix_to_pose3

    device = "cuda"
    H, W = 64, 64
    n_samples = 128
    K = torch.tensor([[50, 0, 32], [0, 50, 32], [0, 0, 1]], dtype=torch.float32, device=device)

    np.random.seed(42)
    gmap = _make_scene(device)

    # Render target from a slightly perturbed pose
    pose_np = np.eye(4)
    pose_np[0, 3] = 0.1  # small offset so gradients are nonzero
    viewmat = torch.inverse(torch.tensor(pose_np, dtype=torch.float32, device=device))
    with torch.no_grad():
        target, _, _ = render_gaussians(
            gmap.means, gmap.quats, gmap.scales, gmap.opacities, gmap.colors,
            viewmat, K, W, H,
        )

    pixel_idx = sample_pixel_indices(H, W, n_samples, rng=np.random.default_rng(0))
    pose3 = matrix_to_pose3(np.eye(4))  # evaluate at identity (not the target pose)

    # Numerical
    factor_num = GaussianSplatFactor(gmap, target, K, pixel_idx, W, H, device, use_analytical=False)
    res_num, J_num = factor_num.evaluate(pose3)

    # Analytical
    factor_ana = GaussianSplatFactor(gmap, target, K, pixel_idx, W, H, device, use_analytical=True)
    res_ana, J_ana = factor_ana.evaluate(pose3)

    # Residuals should match exactly
    np.testing.assert_allclose(res_ana, res_num, atol=1e-5,
                               err_msg="Residuals differ between analytical and numerical")

    # Jacobians should match within numerical tolerance
    # Normalize by max abs value to get relative error
    J_scale = max(np.abs(J_num).max(), 1e-8)
    np.testing.assert_allclose(J_ana / J_scale, J_num / J_scale, atol=0.05,
                               err_msg="Analytical Jacobian deviates from numerical by >5%")


@requires_all
def test_analytical_jacobian_descent_direction():
    """Verify that J^T @ r points in a descent direction (error decreases)."""
    from gsplat_slam.factor import GaussianSplatFactor
    from gsplat_slam.renderer import render_gaussians, sample_pixel_indices
    from gsplat_slam.pose_utils import matrix_to_pose3

    device = "cuda"
    H, W = 64, 64
    n_samples = 256
    K = torch.tensor([[50, 0, 32], [0, 50, 32], [0, 0, 1]], dtype=torch.float32, device=device)

    np.random.seed(123)
    gmap = _make_scene(device)

    # Render target from ground truth
    gt_pose = np.eye(4)
    viewmat = torch.eye(4, dtype=torch.float32, device=device)
    with torch.no_grad():
        target, _, _ = render_gaussians(
            gmap.means, gmap.quats, gmap.scales, gmap.opacities, gmap.colors,
            viewmat, K, W, H,
        )

    pixel_idx = sample_pixel_indices(H, W, n_samples, rng=np.random.default_rng(1))

    # Evaluate at a perturbed pose
    perturbed = np.eye(4)
    perturbed[0, 3] = 0.3
    pose3 = matrix_to_pose3(perturbed)

    factor = GaussianSplatFactor(gmap, target, K, pixel_idx, W, H, device, use_analytical=True)
    residual, J = factor.evaluate(pose3)

    # Gauss-Newton step: delta = -(J^T J)^{-1} J^T r
    JtJ = J.T @ J
    Jtr = J.T @ residual
    delta = -np.linalg.solve(JtJ + 1e-6 * np.eye(6), Jtr)

    # Take a small step and verify error decreases
    new_pose = pose3.retract(delta * 0.1)
    err_before = factor.error_at(pose3)
    err_after = factor.error_at(new_pose)
    assert err_after < err_before, (
        f"GN step should reduce error: {err_before:.6f} -> {err_after:.6f}"
    )
