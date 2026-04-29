"""Tests for GaussianSplatFactor. GPU tests require CUDA + gsplat."""

import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

try:
    import gtsam
    HAS_GTSAM = True
except ImportError:
    HAS_GTSAM = False

try:
    from gsplat import rasterization
    HAS_GSPLAT = True
except ImportError:
    HAS_GSPLAT = False


requires_cuda_gsplat = pytest.mark.skipif(
    not (HAS_CUDA and HAS_GSPLAT and HAS_GTSAM),
    reason="Requires CUDA + gsplat + gtsam"
)


def _make_visible_scene(device="cuda"):
    """Create a Gaussian scene that renders visible content from z=5."""
    from gsplat_slam.map import GaussianMap
    n = 500
    # Place Gaussians in a 2m x 2m plane at z=0, visible from z=5
    means = np.zeros((n, 3), dtype=np.float32)
    means[:, 0] = np.random.uniform(-1, 1, n)
    means[:, 1] = np.random.uniform(-1, 1, n)
    means[:, 2] = np.random.uniform(-0.5, 0.5, n)
    colors = np.random.rand(n, 3).astype(np.float32)
    gmap = GaussianMap.from_pointcloud(means, colors, device=device)
    # Use larger scales so they're visible
    gmap.scales.data.fill_(-1.5)  # exp(-1.5) ~ 0.22, visible blobs
    gmap.opacities.data.fill_(5.0)  # sigmoid(5) ~ 0.99, fully opaque
    return gmap


@requires_cuda_gsplat
def test_splat_factor_residual_at_correct_pose():
    """If we render from the ground-truth pose, residual should be near zero."""
    from gsplat_slam.factor import GaussianSplatFactor
    from gsplat_slam.renderer import render_gaussians, sample_pixel_indices
    from gsplat_slam.pose_utils import matrix_to_pose3

    device = "cuda"
    H, W = 64, 64
    K = torch.tensor([[50, 0, 32], [0, 50, 32], [0, 0, 1]], dtype=torch.float32, device=device)
    gmap = _make_visible_scene(device)

    gt_pose = np.eye(4)
    gt_pose[2, 3] = 5.0

    viewmat = torch.inverse(torch.tensor(gt_pose, dtype=torch.float32, device=device))
    with torch.no_grad():
        target, _, _ = render_gaussians(
            gmap.means, gmap.quats, gmap.scales, gmap.opacities, gmap.colors,
            viewmat, K, W, H,
        )

    pixel_idx = sample_pixel_indices(H, W, 256)
    factor = GaussianSplatFactor(gmap, target, K, pixel_idx, W, H, device)

    err = factor.error_at(matrix_to_pose3(gt_pose))
    assert err < 1e-6, f"Error at ground truth pose should be ~0, got {err}"


@requires_cuda_gsplat
def test_splat_factor_error_increases_with_perturbation():
    """Perturbing the pose should increase the rendering error."""
    from gsplat_slam.factor import GaussianSplatFactor
    from gsplat_slam.renderer import render_gaussians, sample_pixel_indices
    from gsplat_slam.pose_utils import matrix_to_pose3

    device = "cuda"
    H, W = 64, 64
    K = torch.tensor([[50, 0, 32], [0, 50, 32], [0, 0, 1]], dtype=torch.float32, device=device)
    gmap = _make_visible_scene(device)

    gt_pose = np.eye(4)
    gt_pose[2, 3] = 5.0

    viewmat = torch.inverse(torch.tensor(gt_pose, dtype=torch.float32, device=device))
    with torch.no_grad():
        target, _, _ = render_gaussians(
            gmap.means, gmap.quats, gmap.scales, gmap.opacities, gmap.colors,
            viewmat, K, W, H,
        )

    pixel_idx = sample_pixel_indices(H, W, 256)
    factor = GaussianSplatFactor(gmap, target, K, pixel_idx, W, H, device)

    err_gt = factor.error_at(matrix_to_pose3(gt_pose))

    perturbed = np.eye(4)
    perturbed[2, 3] = 5.5  # 0.5m off
    err_perturbed = factor.error_at(matrix_to_pose3(perturbed))

    assert err_perturbed > err_gt, (
        f"Perturbed error ({err_perturbed}) should exceed GT error ({err_gt})"
    )


@requires_cuda_gsplat
def test_splat_factor_jacobian_shape():
    """Check that the Jacobian has the right shape."""
    from gsplat_slam.factor import GaussianSplatFactor
    from gsplat_slam.renderer import render_gaussians, sample_pixel_indices
    from gsplat_slam.pose_utils import matrix_to_pose3

    device = "cuda"
    H, W = 32, 32
    n_samples = 64
    K = torch.tensor([[50, 0, 16], [0, 50, 16], [0, 0, 1]], dtype=torch.float32, device=device)
    gmap = _make_visible_scene(device)

    gt_pose = np.eye(4)
    gt_pose[2, 3] = 5.0
    viewmat = torch.inverse(torch.tensor(gt_pose, dtype=torch.float32, device=device))
    with torch.no_grad():
        target, _, _ = render_gaussians(
            gmap.means, gmap.quats, gmap.scales, gmap.opacities, gmap.colors,
            viewmat, K, W, H,
        )

    pixel_idx = sample_pixel_indices(H, W, n_samples)
    factor = GaussianSplatFactor(gmap, target, K, pixel_idx, W, H, device)

    residual, J = factor.evaluate(matrix_to_pose3(gt_pose))

    assert residual.shape == (n_samples * 3,)
    assert J.shape == (n_samples * 3, 6)
