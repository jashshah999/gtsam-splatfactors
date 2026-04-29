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


@requires_cuda_gsplat
def test_splat_factor_residual_at_correct_pose():
    """If we render from the ground-truth pose, residual should be near zero."""
    from gsplat_slam.factor import GaussianSplatFactor
    from gsplat_slam.map import GaussianMap
    from gsplat_slam.renderer import render_gaussians, sample_pixel_indices
    from gsplat_slam.pose_utils import matrix_to_pose3

    device = "cuda"
    H, W = 64, 64
    K = torch.tensor([[50, 0, 32], [0, 50, 32], [0, 0, 1]], dtype=torch.float32, device=device)

    gmap = GaussianMap.from_pointcloud(
        np.random.randn(200, 3).astype(np.float32) * 2,
        np.random.rand(200, 3).astype(np.float32),
        device=device,
    )

    gt_pose = np.eye(4)
    gt_pose[2, 3] = 5.0  # camera 5m back

    viewmat = torch.inverse(
        torch.tensor(gt_pose, dtype=torch.float32, device=device)
    )
    with torch.no_grad():
        target, _, _ = render_gaussians(
            gmap.means, gmap.quats, gmap.scales, gmap.opacities, gmap.colors,
            viewmat, K, W, H,
        )

    pixel_idx = sample_pixel_indices(H, W, 256)
    factor = GaussianSplatFactor(gmap, target, K, pixel_idx, W, H, device)

    pose = matrix_to_pose3(gt_pose)
    err = factor.error_at(pose)
    assert err < 1e-6, f"Error at ground truth pose should be ~0, got {err}"


@requires_cuda_gsplat
def test_splat_factor_error_increases_with_perturbation():
    """Perturbing the pose should increase the rendering error."""
    from gsplat_slam.factor import GaussianSplatFactor
    from gsplat_slam.map import GaussianMap
    from gsplat_slam.renderer import render_gaussians, sample_pixel_indices
    from gsplat_slam.pose_utils import matrix_to_pose3

    device = "cuda"
    H, W = 64, 64
    K = torch.tensor([[50, 0, 32], [0, 50, 32], [0, 0, 1]], dtype=torch.float32, device=device)

    gmap = GaussianMap.from_pointcloud(
        np.random.randn(200, 3).astype(np.float32) * 2,
        np.random.rand(200, 3).astype(np.float32),
        device=device,
    )

    gt_pose = np.eye(4)
    gt_pose[2, 3] = 5.0

    viewmat = torch.inverse(
        torch.tensor(gt_pose, dtype=torch.float32, device=device)
    )
    with torch.no_grad():
        target, _, _ = render_gaussians(
            gmap.means, gmap.quats, gmap.scales, gmap.opacities, gmap.colors,
            viewmat, K, W, H,
        )

    pixel_idx = sample_pixel_indices(H, W, 256)
    factor = GaussianSplatFactor(gmap, target, K, pixel_idx, W, H, device)

    pose_gt = matrix_to_pose3(gt_pose)
    err_gt = factor.error_at(pose_gt)

    perturbed = np.eye(4)
    perturbed[2, 3] = 5.3  # 0.3m off
    pose_perturbed = matrix_to_pose3(perturbed)
    err_perturbed = factor.error_at(pose_perturbed)

    assert err_perturbed > err_gt, (
        f"Perturbed error ({err_perturbed}) should exceed GT error ({err_gt})"
    )


@requires_cuda_gsplat
def test_splat_factor_jacobian_shape():
    """Check that the Jacobian has the right shape."""
    from gsplat_slam.factor import GaussianSplatFactor
    from gsplat_slam.map import GaussianMap
    from gsplat_slam.renderer import render_gaussians, sample_pixel_indices
    from gsplat_slam.pose_utils import matrix_to_pose3

    device = "cuda"
    H, W = 32, 32
    n_samples = 64
    K = torch.tensor([[50, 0, 16], [0, 50, 16], [0, 0, 1]], dtype=torch.float32, device=device)

    gmap = GaussianMap.from_pointcloud(
        np.random.randn(100, 3).astype(np.float32) * 2,
        device=device,
    )

    gt_pose = np.eye(4)
    gt_pose[2, 3] = 5.0
    viewmat = torch.inverse(
        torch.tensor(gt_pose, dtype=torch.float32, device=device)
    )
    with torch.no_grad():
        target, _, _ = render_gaussians(
            gmap.means, gmap.quats, gmap.scales, gmap.opacities, gmap.colors,
            viewmat, K, W, H,
        )

    pixel_idx = sample_pixel_indices(H, W, n_samples)
    factor = GaussianSplatFactor(gmap, target, K, pixel_idx, W, H, device)

    pose = matrix_to_pose3(gt_pose)
    residual, J = factor.evaluate(pose)

    assert residual.shape == (n_samples * 3,)
    assert J.shape == (n_samples * 3, 6)
