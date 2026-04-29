"""Synthetic demo: build a Gaussian scene, render keyframes along a circular
trajectory, then run SplatSLAM to recover the poses and close the loop.

Usage:
    python examples/synthetic_demo.py

Requires: CUDA GPU, gsplat, gtsam, torch, opencv-python
"""

import numpy as np
import torch
import cv2
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gsplat_slam.map import GaussianMap
from gsplat_slam.renderer import render_gaussians
from gsplat_slam.slam import SplatSLAM


def make_synthetic_scene(device="cuda", n_gaussians=2000):
    """Create a colorful room-like Gaussian scene."""
    means = np.zeros((n_gaussians, 3), dtype=np.float32)
    colors = np.zeros((n_gaussians, 3), dtype=np.float32)

    n_per_surface = n_gaussians // 5

    # Floor (y = -1)
    idx = 0
    means[idx:idx+n_per_surface, 0] = np.random.uniform(-3, 3, n_per_surface)
    means[idx:idx+n_per_surface, 1] = -1 + np.random.normal(0, 0.05, n_per_surface)
    means[idx:idx+n_per_surface, 2] = np.random.uniform(2, 8, n_per_surface)
    colors[idx:idx+n_per_surface] = [0.6, 0.5, 0.4]
    idx += n_per_surface

    # Back wall (z = 8)
    means[idx:idx+n_per_surface, 0] = np.random.uniform(-3, 3, n_per_surface)
    means[idx:idx+n_per_surface, 1] = np.random.uniform(-1, 2, n_per_surface)
    means[idx:idx+n_per_surface, 2] = 8 + np.random.normal(0, 0.05, n_per_surface)
    colors[idx:idx+n_per_surface] = [0.3, 0.4, 0.7]
    idx += n_per_surface

    # Left wall (x = -3)
    means[idx:idx+n_per_surface, 0] = -3 + np.random.normal(0, 0.05, n_per_surface)
    means[idx:idx+n_per_surface, 1] = np.random.uniform(-1, 2, n_per_surface)
    means[idx:idx+n_per_surface, 2] = np.random.uniform(2, 8, n_per_surface)
    colors[idx:idx+n_per_surface] = [0.7, 0.3, 0.3]
    idx += n_per_surface

    # Right wall (x = 3)
    means[idx:idx+n_per_surface, 0] = 3 + np.random.normal(0, 0.05, n_per_surface)
    means[idx:idx+n_per_surface, 1] = np.random.uniform(-1, 2, n_per_surface)
    means[idx:idx+n_per_surface, 2] = np.random.uniform(2, 8, n_per_surface)
    colors[idx:idx+n_per_surface] = [0.3, 0.7, 0.3]
    idx += n_per_surface

    # Random objects
    remaining = n_gaussians - idx
    means[idx:, 0] = np.random.uniform(-2, 2, remaining)
    means[idx:, 1] = np.random.uniform(-0.5, 1.5, remaining)
    means[idx:, 2] = np.random.uniform(3, 7, remaining)
    colors[idx:] = np.random.rand(remaining, 3)

    gmap = GaussianMap.from_pointcloud(means, colors, device=device)
    gmap.scales.data.fill_(-1.0)
    gmap.opacities.data.fill_(0.95)
    return gmap


def make_circular_trajectory(n_frames=20, radius=1.5, center_z=5.0, height=0.5):
    """Generate camera poses on a circle looking inward."""
    poses = []
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        x = radius * np.cos(angle)
        z = center_z + radius * np.sin(angle)

        # Camera looks toward center of circle
        forward = np.array([0, 0, center_z]) - np.array([x, height, z])
        forward = forward / np.linalg.norm(forward)
        right = np.cross(np.array([0, 1, 0]), forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)

        T = np.eye(4)
        T[:3, 0] = right
        T[:3, 1] = up
        T[:3, 2] = forward
        T[:3, 3] = [x, height, z]
        poses.append(T)
    return poses


def render_from_pose(gmap, pose, K, W, H, device):
    """Render the Gaussian map from a world-to-camera pose."""
    T = torch.tensor(pose, dtype=torch.float32, device=device)
    viewmat = torch.inverse(T)
    with torch.no_grad():
        rendered, _, _ = render_gaussians(
            gmap.means, gmap.quats, gmap.scales, gmap.opacities, gmap.colors,
            viewmat, K, W, H,
        )
    return rendered.cpu().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No CUDA GPU found. This demo requires a GPU.")
        return

    print("=== gtsam-splatfactors Synthetic Demo ===\n")

    # Setup
    H, W = 120, 160
    K_np = np.array([[80, 0, 80], [0, 80, 60], [0, 0, 1]], dtype=np.float64)
    K_torch = torch.tensor(K_np, dtype=torch.float32, device=device)

    # Create scene
    print("Creating synthetic Gaussian scene...")
    gt_map = make_synthetic_scene(device, n_gaussians=3000)
    print(f"  {gt_map.n_gaussians} Gaussians\n")

    # Generate trajectory
    n_frames = 16
    gt_poses = make_circular_trajectory(n_frames, radius=1.5, center_z=5.0)

    # Render ground-truth images
    print("Rendering ground-truth keyframes...")
    images = []
    for i, pose in enumerate(gt_poses):
        img = render_from_pose(gt_map, pose, K_torch, W, H, device)
        images.append(img)
    print(f"  {len(images)} frames rendered\n")

    # Save a few sample images
    os.makedirs("output", exist_ok=True)
    for i in [0, n_frames//4, n_frames//2]:
        img_uint8 = (np.clip(images[i], 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(f"output/gt_frame_{i:03d}.png", cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
    print("  Saved sample frames to output/\n")

    # Run SLAM with noisy initial poses
    print("Running SplatSLAM...")
    import gtsam
    params = gtsam.ISAM2Params()
    slam = SplatSLAM(
        K=K_np, W=W, H=H,
        n_pixel_samples=512,
        device=device,
        isam2_params=params,
        mapping_iters=0,  # skip mapping, use GT Gaussians
        photo_sigma=0.5,  # per-pixel photometric noise
        odom_sigma=0.15,  # odometry noise (we add ~0.1m perturbation)
        tracking_iters=5, # iterate iSAM2 for convergence
    )
    # Inject the ground-truth Gaussians so tracking uses them
    slam.gaussian_map = gt_map

    estimated_poses = []
    for i in range(n_frames):
        # Add noise to the initial guess
        noise = np.eye(4)
        noise[:3, 3] += np.random.randn(3) * 0.1
        init_pose = gt_poses[i] @ noise

        est_pose = slam.add_keyframe(
            images[i],
            init_pose=init_pose,
            prior_sigma=0.5 if i == 0 else None,
        )
        estimated_poses.append(est_pose)

        gt_t = gt_poses[i][:3, 3]
        est_t = est_pose[:3, 3]
        err = np.linalg.norm(gt_t - est_t)
        print(f"  Frame {i:2d}: translation error = {err:.4f} m")

    # Add loop closure (last frame sees same scene as first frame)
    print("\nAdding loop closure (frame 0 <-> frame {})...".format(n_frames - 1))
    relative = np.linalg.inv(gt_poses[0]) @ gt_poses[n_frames - 1]
    slam.add_loop_closure(0, n_frames - 1, relative, sigma=0.01)

    # Get corrected poses
    corrected = slam.get_all_poses()
    print("\nAfter loop closure:")
    total_err = 0
    for i in range(n_frames):
        gt_t = gt_poses[i][:3, 3]
        est_t = corrected[i][:3, 3]
        err = np.linalg.norm(gt_t - est_t)
        total_err += err
        print(f"  Frame {i:2d}: translation error = {err:.4f} m")

    ate = total_err / n_frames
    print(f"\nAverage Translation Error (ATE): {ate:.4f} m")
    print("\nDone!")


if __name__ == "__main__":
    main()
