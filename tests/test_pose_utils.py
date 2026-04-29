"""Tests for pose conversion utilities. Runs without CUDA."""

import numpy as np
import pytest

try:
    import gtsam
    HAS_GTSAM = True
except ImportError:
    HAS_GTSAM = False

pytestmark = pytest.mark.skipif(not HAS_GTSAM, reason="gtsam not installed")


def test_pose3_roundtrip():
    from gsplat_slam.pose_utils import pose3_to_matrix, matrix_to_pose3

    R = gtsam.Rot3.Rodrigues(0.1, 0.2, 0.3)
    t = gtsam.Point3(1.0, 2.0, 3.0)
    pose = gtsam.Pose3(R, t)

    T = pose3_to_matrix(pose)
    assert T.shape == (4, 4)
    assert np.allclose(T[3, :], [0, 0, 0, 1])

    pose2 = matrix_to_pose3(T)
    assert pose.equals(pose2, 1e-9)


def test_identity_pose():
    from gsplat_slam.pose_utils import pose3_to_matrix, matrix_to_pose3

    pose = gtsam.Pose3()
    T = pose3_to_matrix(pose)
    assert np.allclose(T, np.eye(4), atol=1e-12)

    pose2 = matrix_to_pose3(np.eye(4))
    assert pose.equals(pose2, 1e-9)


def test_numeric_pose_jacobian():
    from gsplat_slam.pose_utils import numeric_pose_jacobian, pose3_to_matrix

    def translation_func(pose):
        return pose3_to_matrix(pose)[:3, 3]

    pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 2, 3))
    J = numeric_pose_jacobian(translation_func, pose)

    assert J.shape == (3, 6)
    # Translation is affected by rotation (cross product) and direct translation
    assert not np.allclose(J, 0)


def test_sample_pixel_indices():
    from gsplat_slam.renderer import sample_pixel_indices

    idx = sample_pixel_indices(480, 640, 100, rng=np.random.default_rng(0))
    assert idx.shape == (100, 2)
    assert np.all(idx[:, 0] >= 0) and np.all(idx[:, 0] < 480)
    assert np.all(idx[:, 1] >= 0) and np.all(idx[:, 1] < 640)
    # No duplicates
    flat = idx[:, 0] * 640 + idx[:, 1]
    assert len(np.unique(flat)) == 100
