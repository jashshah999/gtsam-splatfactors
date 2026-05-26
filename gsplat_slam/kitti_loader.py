"""KITTI Odometry dataset loader.

Loads sequences from the KITTI Visual Odometry benchmark.
Expects data at: data_root/sequences/{seq_id:02d}/

Structure:
    sequences/00/image_2/000000.png  (left color camera)
    sequences/00/image_3/000000.png  (right color camera)
    poses/00.txt                     (ground truth poses, 12 values per line)
    sequences/00/calib.txt           (calibration)

Download from: https://www.cvlibs.net/datasets/kitti/eval_odometry.php
"""

import os
import numpy as np
import cv2


KITTI_SEQUENCES_WITH_GT = list(range(11))  # 00-10 have ground truth


def read_kitti_poses(filepath: str) -> list:
    """Read KITTI ground truth poses (3x4 matrix per line)."""
    poses = []
    with open(filepath) as f:
        for line in f:
            values = [float(v) for v in line.strip().split()]
            if len(values) != 12:
                continue
            T = np.eye(4)
            T[:3, :] = np.array(values).reshape(3, 4)
            poses.append(T)
    return poses


def read_kitti_calib(filepath: str) -> dict:
    """Read KITTI calibration file."""
    calib = {}
    with open(filepath) as f:
        for line in f:
            if ':' not in line:
                continue
            key, values = line.split(':', 1)
            calib[key.strip()] = np.array([float(v) for v in values.strip().split()])
    return calib


def get_kitti_intrinsics(calib_path: str, camera: str = None) -> np.ndarray:
    """Get 3x3 intrinsics matrix. Uses P0 for grayscale, P2 for color."""
    calib = read_kitti_calib(calib_path)
    if camera and camera.startswith("image_2"):
        key = "P2"
    elif "P0" in calib:
        key = "P0"
    else:
        key = "P2"
    P = calib[key].reshape(3, 4)
    K = P[:3, :3]
    return K


class KITTIDataset:
    """Load a KITTI Odometry sequence."""

    def __init__(
        self,
        seq_id: int = 0,
        data_root: str = "data/kitti",
        stride: int = 1,
        max_frames: int = -1,
        image_cam: str = None,
    ):
        assert 0 <= seq_id <= 10, f"Only sequences 00-10 have GT. Got {seq_id}"
        self.seq_id = seq_id
        self.data_root = data_root
        self.stride = stride

        seq_dir = os.path.join(data_root, "sequences", f"{seq_id:02d}")
        assert os.path.isdir(seq_dir), (
            f"KITTI sequence not found at {seq_dir}. "
            f"Download from https://www.cvlibs.net/datasets/kitti/eval_odometry.php"
        )

        # Auto-detect: prefer color (image_2), fall back to grayscale (image_0)
        if image_cam is None:
            if os.path.isdir(os.path.join(seq_dir, "image_2")):
                image_cam = "image_2"
            else:
                image_cam = "image_0"
        self.image_cam = image_cam

        # Calibration
        calib_path = os.path.join(seq_dir, "calib.txt")
        self.K = get_kitti_intrinsics(calib_path, camera=image_cam)

        # Poses
        pose_path = os.path.join(data_root, "poses", f"{seq_id:02d}.txt")
        assert os.path.exists(pose_path), f"Pose file not found: {pose_path}"
        self.poses = read_kitti_poses(pose_path)

        # Images
        img_dir = os.path.join(seq_dir, image_cam)
        self.image_files = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.endswith(".png")
        ])

        # Match frames to poses
        n_frames = min(len(self.image_files), len(self.poses))
        self.indices = list(range(0, n_frames, stride))
        if max_frames > 0:
            self.indices = self.indices[:max_frames]

        # Image dimensions (from first frame)
        sample = cv2.imread(self.image_files[0])
        self.H, self.W = sample.shape[:2]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        frame_idx = self.indices[idx]
        img = cv2.imread(self.image_files[frame_idx])
        if img.ndim == 2 or img.shape[2] == 1:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return {
            "rgb": rgb,
            "depth": None,
            "pose": self.poses[frame_idx],
            "frame_id": frame_idx,
        }


def compute_stereo_depth(
    left_path: str,
    right_path: str,
    K: np.ndarray,
    baseline: float = 0.54,
) -> np.ndarray:
    """Compute depth from stereo pair using StereoSGBM."""
    left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=5,
        P1=8 * 1 * 5 * 5,
        P2=32 * 1 * 5 * 5,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )
    disparity = stereo.compute(left, right).astype(np.float32) / 16.0
    valid = disparity > 0
    depth = np.zeros_like(disparity)
    depth[valid] = K[0, 0] * baseline / disparity[valid]
    return depth


class KITTIStereoDataset(KITTIDataset):
    """KITTI with stereo depth computation."""

    def __init__(self, baseline: float = 0.54, **kwargs):
        super().__init__(**kwargs)
        self.baseline = baseline
        seq_dir = os.path.join(self.data_root, "sequences", f"{self.seq_id:02d}")
        # Try color pair (image_2/image_3) first, fall back to grayscale (image_0/image_1)
        right_color = os.path.join(seq_dir, "image_3")
        right_gray = os.path.join(seq_dir, "image_1")
        self.right_dir = right_color if os.path.isdir(right_color) else right_gray

    def __getitem__(self, idx):
        frame = super().__getitem__(idx)
        frame_idx = self.indices[idx]
        left_path = self.image_files[frame_idx]
        right_path = os.path.join(self.right_dir, os.path.basename(left_path))

        if os.path.exists(right_path):
            frame["depth"] = compute_stereo_depth(
                left_path, right_path, self.K, self.baseline
            )
        return frame
