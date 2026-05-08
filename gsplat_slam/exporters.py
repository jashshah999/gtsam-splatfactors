"""Export Gaussian map and poses to standard formats."""

import numpy as np
import os
import json
import struct
from pathlib import Path
from scipy.spatial.transform import Rotation


def export_all(slam, output_dir: str, image_paths: list[str] = None):
    """Export everything from a SplatSLAM instance."""
    os.makedirs(output_dir, exist_ok=True)
    poses = slam.get_all_poses()
    K = slam.K_np

    pose_array = np.array([poses[i] for i in sorted(poses.keys())])
    n = len(pose_array)
    intrinsics = np.tile(K, (n, 1, 1))

    if image_paths is None:
        image_paths = [f"frame_{i:04d}.jpg" for i in range(n)]

    # Export poses
    np.save(os.path.join(output_dir, "poses_c2w.npy"), pose_array)

    # COLMAP format
    export_colmap(pose_array, intrinsics, image_paths, output_dir, slam.W, slam.H)

    # nerfstudio format
    export_nerfstudio(pose_array, K, image_paths, output_dir, slam.W, slam.H)

    # Gaussian map as PLY
    if slam.gaussian_map.n_gaussians > 0:
        export_gaussians_ply(slam.gaussian_map, os.path.join(output_dir, "gaussians.ply"))

    print(f"Exported to {output_dir}/")


def export_colmap(poses_c2w, intrinsics, image_paths, output_dir, W, H):
    """Export to COLMAP sparse text format."""
    sparse_dir = Path(output_dir) / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    N = len(poses_c2w)

    with open(sparse_dir / "cameras.txt", "w") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        K = intrinsics[0]
        f.write(f"1 PINHOLE {W} {H} {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}\n")

    with open(sparse_dir / "images.txt", "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for i in range(N):
            w2c = np.linalg.inv(poses_c2w[i])
            quat = Rotation.from_matrix(w2c[:3, :3]).as_quat()
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            t = w2c[:3, 3]
            name = os.path.basename(image_paths[i]) if i < len(image_paths) else f"frame_{i:04d}.jpg"
            f.write(f"{i+1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {name}\n\n")

    with open(sparse_dir / "points3D.txt", "w") as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")

    print(f"  COLMAP: {sparse_dir}")


def export_nerfstudio(poses_c2w, K, image_paths, output_dir, W, H):
    """Export to nerfstudio transforms.json."""
    frames = []
    for i in range(len(poses_c2w)):
        c2w = poses_c2w[i].copy()
        c2w[:3, 1:3] *= -1
        frames.append({
            "file_path": image_paths[i] if i < len(image_paths) else f"frame_{i:04d}.jpg",
            "transform_matrix": c2w.tolist(),
        })

    data = {
        "fl_x": float(K[0, 0]), "fl_y": float(K[1, 1]),
        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "w": W, "h": H, "aabb_scale": 16,
        "frames": frames,
    }

    path = os.path.join(output_dir, "transforms.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  nerfstudio: {path}")


def export_gaussians_ply(gaussian_map, output_path: str):
    """Export Gaussian map as a colored PLY point cloud."""
    import torch

    means = gaussian_map.means.detach().cpu().numpy()
    colors = gaussian_map.colors.detach().cpu().numpy()
    colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    N = len(means)

    with open(output_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            f.write(f"{means[i,0]} {means[i,1]} {means[i,2]} {colors[i,0]} {colors[i,1]} {colors[i,2]}\n")

    print(f"  Gaussians PLY: {output_path} ({N} gaussians)")
