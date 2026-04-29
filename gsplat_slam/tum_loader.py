"""TUM-RGBD dataset loader.

Downloads and loads sequences from the TUM RGB-D benchmark:
https://cvg.cit.tum.de/data/datasets/rgbd-dataset

Usage:
    loader = TUMDataset("fr1/desk")
    for rgb, depth, pose, timestamp in loader:
        ...
"""

import os
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

TUM_BASE_URL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg{fid}/rgbd_dataset_freiburg{fid}_{seq_name}.tgz"

SEQUENCES = {
    "fr1/desk": ("1", "desk"),
    "fr1/room": ("1", "room"),
    "fr1/xyz": ("1", "xyz"),
    "fr1/360": ("1", "360"),
    "fr2/desk": ("2", "desk"),
    "fr2/xyz": ("2", "xyz"),
    "fr3/office": ("3", "long_office_household"),
}

# TUM-RGBD camera intrinsics (Freiburg 1)
TUM_FR1_K = np.array([
    [517.3, 0, 318.6],
    [0, 516.5, 255.3],
    [0, 0, 1],
], dtype=np.float64)

TUM_FR2_K = np.array([
    [520.9, 0, 325.1],
    [0, 521.0, 249.7],
    [0, 0, 1],
], dtype=np.float64)

TUM_FR3_K = np.array([
    [535.4, 0, 320.1],
    [0, 539.2, 247.6],
    [0, 0, 1],
], dtype=np.float64)


def get_intrinsics(seq_name: str) -> np.ndarray:
    if seq_name.startswith("fr1"):
        return TUM_FR1_K
    elif seq_name.startswith("fr2"):
        return TUM_FR2_K
    else:
        return TUM_FR3_K


def quaternion_to_matrix(qx, qy, qz, qw) -> np.ndarray:
    """Convert quaternion (qx,qy,qz,qw) to 3x3 rotation matrix."""
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])
    return R


def read_trajectory(filepath: str) -> dict:
    """Read TUM groundtruth.txt: timestamp tx ty tz qx qy qz qw -> {timestamp: 4x4 matrix}."""
    poses = {}
    with open(filepath) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            ts = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            T = np.eye(4)
            T[:3, :3] = quaternion_to_matrix(qx, qy, qz, qw)
            T[:3, 3] = [tx, ty, tz]
            poses[ts] = T
    return poses


def read_associations(filepath: str) -> list:
    """Read associations.txt: ts_rgb path_rgb ts_depth path_depth."""
    assoc = []
    with open(filepath) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 4:
                assoc.append({
                    "ts_rgb": float(parts[0]),
                    "rgb_path": parts[1],
                    "ts_depth": float(parts[2]),
                    "depth_path": parts[3],
                })
    return assoc


def associate_with_gt(associations: list, gt_poses: dict, max_dt: float = 0.02) -> list:
    """Match each RGB-D frame to the closest groundtruth pose."""
    gt_times = sorted(gt_poses.keys())
    result = []
    for a in associations:
        ts = a["ts_rgb"]
        idx = np.searchsorted(gt_times, ts)
        best_dt = float("inf")
        best_ts = None
        for i in [max(0, idx - 1), min(len(gt_times) - 1, idx)]:
            dt = abs(gt_times[i] - ts)
            if dt < best_dt:
                best_dt = dt
                best_ts = gt_times[i]
        if best_dt < max_dt:
            a["gt_pose"] = gt_poses[best_ts]
            a["gt_timestamp"] = best_ts
            result.append(a)
    return result


def generate_associations(data_dir: str) -> str:
    """Generate associations file using associate.py logic."""
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")

    rgb_ts = {}
    for f in sorted(os.listdir(rgb_dir)):
        if f.endswith(".png"):
            ts = float(f.replace(".png", ""))
            rgb_ts[ts] = os.path.join("rgb", f)

    depth_ts = {}
    for f in sorted(os.listdir(depth_dir)):
        if f.endswith(".png"):
            ts = float(f.replace(".png", ""))
            depth_ts[ts] = os.path.join("depth", f)

    assoc_path = os.path.join(data_dir, "associations.txt")
    rgb_keys = sorted(rgb_ts.keys())
    depth_keys = sorted(depth_ts.keys())

    with open(assoc_path, "w") as out:
        for rt in rgb_keys:
            idx = np.searchsorted(depth_keys, rt)
            best_dt = float("inf")
            best_dt_key = None
            for i in [max(0, idx - 1), min(len(depth_keys) - 1, idx)]:
                dt = abs(depth_keys[i] - rt)
                if dt < best_dt:
                    best_dt = dt
                    best_dt_key = depth_keys[i]
            if best_dt < 0.02:
                out.write(f"{rt:.6f} {rgb_ts[rt]} {best_dt_key:.6f} {depth_ts[best_dt_key]}\n")

    return assoc_path


class TUMDataset:
    """Load a TUM-RGBD sequence."""

    def __init__(self, seq_name: str, data_root: str = "data", stride: int = 1,
                 max_frames: int = -1):
        assert seq_name in SEQUENCES, f"Unknown sequence: {seq_name}. Options: {list(SEQUENCES.keys())}"
        self.seq_name = seq_name
        self.K = get_intrinsics(seq_name)
        self.H, self.W = 480, 640
        self.stride = stride

        fid, sname = SEQUENCES[seq_name]
        self.data_dir = os.path.join(data_root, f"rgbd_dataset_freiburg{fid}_{sname}")

        if not os.path.isdir(self.data_dir):
            self._download(data_root, fid, sname)

        gt_path = os.path.join(self.data_dir, "groundtruth.txt")
        gt_poses = read_trajectory(gt_path)

        assoc_path = os.path.join(self.data_dir, "associations.txt")
        if not os.path.exists(assoc_path):
            assoc_path = generate_associations(self.data_dir)

        associations = read_associations(assoc_path)
        self.frames = associate_with_gt(associations, gt_poses)
        self.frames = self.frames[::stride]
        if max_frames > 0:
            self.frames = self.frames[:max_frames]

    def _download(self, data_root: str, fid: str, sname: str):
        os.makedirs(data_root, exist_ok=True)
        url = TUM_BASE_URL.format(fid=fid, seq_name=sname)
        tgz_path = os.path.join(data_root, f"freiburg{fid}_{sname}.tgz")
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, tgz_path)
        print(f"Extracting to {data_root}/...")
        with tarfile.open(tgz_path) as tar:
            tar.extractall(data_root)
        os.remove(tgz_path)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        rgb_path = os.path.join(self.data_dir, frame["rgb_path"])
        depth_path = os.path.join(self.data_dir, frame["depth_path"])

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 5000.0  # TUM depth factor

        return {
            "rgb": rgb,
            "depth": depth,
            "pose": frame["gt_pose"],
            "timestamp": frame["ts_rgb"],
        }
