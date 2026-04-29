"""Conversions between GTSAM Pose3 and torch/numpy 4x4 matrices."""

import numpy as np

try:
    import gtsam
    HAS_GTSAM = True
except ImportError:
    HAS_GTSAM = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def pose3_to_matrix(pose) -> np.ndarray:
    """Convert a gtsam.Pose3 to a 4x4 numpy matrix (world-from-camera)."""
    return pose.matrix()


def matrix_to_pose3(T: np.ndarray):
    """Convert a 4x4 numpy matrix to a gtsam.Pose3."""
    assert T.shape == (4, 4), f"Expected 4x4 matrix, got {T.shape}"
    R = gtsam.Rot3(T[:3, :3])
    t = gtsam.Point3(T[:3, 3])
    return gtsam.Pose3(R, t)


def torch_to_gtsam_pose(T_torch):
    """Convert a (4,4) torch tensor to gtsam.Pose3."""
    return matrix_to_pose3(T_torch.detach().cpu().numpy())


def gtsam_to_torch_pose(pose, device="cpu"):
    """Convert a gtsam.Pose3 to a (4,4) torch tensor."""
    T = pose3_to_matrix(pose).astype(np.float32)
    return torch.tensor(T, device=device, dtype=torch.float32)


def pose3_to_Rt(pose):
    """Extract (3,3) rotation and (3,) translation from Pose3.
    Returns numpy arrays. Convention: world_T_camera."""
    T = pose.matrix()
    return T[:3, :3], T[:3, 3]


def numeric_pose_jacobian(func, pose, eps=1e-5):
    """Compute 6-column Jacobian of func(pose) via central differences.

    Perturbs pose in the tangent space (Lie algebra) and measures the
    change in the output. Returns (n, 6) Jacobian where n = len(func(pose)).
    """
    f0 = func(pose)
    n = len(f0)
    J = np.zeros((n, 6))
    for i in range(6):
        delta = np.zeros(6)
        delta[i] = eps
        f_plus = func(pose.retract(delta))
        f_minus = func(pose.retract(-delta))
        J[:, i] = (f_plus - f_minus) / (2 * eps)
    return J
