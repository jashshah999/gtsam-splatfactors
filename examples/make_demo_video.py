"""Generate a visual demo: side-by-side GT vs estimated rendering + trajectory.

Produces:
  output/demo.mp4          — video of GT vs re-rendered from estimated poses
  output/trajectory.png    — bird's-eye trajectory plot
  output/comparison_grid.png — grid of GT vs estimated frames

Usage:
    python examples/make_demo_video.py
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


def make_scene(device, n=3000):
    means = np.zeros((n, 3), dtype=np.float32)
    colors = np.zeros((n, 3), dtype=np.float32)
    per = n // 5
    i = 0
    # Floor
    means[i:i+per, 0] = np.random.uniform(-3, 3, per)
    means[i:i+per, 1] = -1 + np.random.normal(0, 0.05, per)
    means[i:i+per, 2] = np.random.uniform(2, 8, per)
    colors[i:i+per] = [0.6, 0.5, 0.4]; i += per
    # Back wall
    means[i:i+per, 0] = np.random.uniform(-3, 3, per)
    means[i:i+per, 1] = np.random.uniform(-1, 2, per)
    means[i:i+per, 2] = 8 + np.random.normal(0, 0.05, per)
    colors[i:i+per] = [0.3, 0.4, 0.7]; i += per
    # Left wall
    means[i:i+per, 0] = -3 + np.random.normal(0, 0.05, per)
    means[i:i+per, 1] = np.random.uniform(-1, 2, per)
    means[i:i+per, 2] = np.random.uniform(2, 8, per)
    colors[i:i+per] = [0.7, 0.3, 0.3]; i += per
    # Right wall
    means[i:i+per, 0] = 3 + np.random.normal(0, 0.05, per)
    means[i:i+per, 1] = np.random.uniform(-1, 2, per)
    means[i:i+per, 2] = np.random.uniform(2, 8, per)
    colors[i:i+per] = [0.3, 0.7, 0.3]; i += per
    # Objects
    rem = n - i
    means[i:, 0] = np.random.uniform(-2, 2, rem)
    means[i:, 1] = np.random.uniform(-0.5, 1.5, rem)
    means[i:, 2] = np.random.uniform(3, 7, rem)
    colors[i:] = np.random.rand(rem, 3)

    gmap = GaussianMap.from_pointcloud(means, colors, device=device)
    gmap.scales.data.fill_(-1.0)
    gmap.opacities.data.fill_(0.95)
    return gmap


def circular_poses(n, radius=1.5, cz=5.0, h=0.5):
    poses = []
    for i in range(n):
        a = 2 * np.pi * i / n
        x, z = radius * np.cos(a), cz + radius * np.sin(a)
        fwd = np.array([0, 0, cz]) - np.array([x, h, z])
        fwd /= np.linalg.norm(fwd)
        right = np.cross([0, 1, 0], fwd); right /= np.linalg.norm(right)
        up = np.cross(fwd, right)
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, up, fwd, [x, h, z]
        poses.append(T)
    return poses


def render_at(gmap, pose, K, W, H, device):
    T = torch.tensor(pose, dtype=torch.float32, device=device)
    vm = torch.inverse(T)
    with torch.no_grad():
        r, _, _ = render_gaussians(
            gmap.means, gmap.quats, gmap.scales, gmap.opacities, gmap.colors,
            vm, K, W, H)
    return np.clip(r.cpu().numpy(), 0, 1)


def to_uint8(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def add_text(img, text, pos=(10, 25), color=(255, 255, 255)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    return img


def main():
    device = "cuda"
    H, W = 240, 320
    K_np = np.array([[160, 0, 160], [0, 160, 120], [0, 0, 1]], dtype=np.float64)
    K_torch = torch.tensor(K_np, dtype=torch.float32, device=device)
    n_frames = 24

    os.makedirs("output", exist_ok=True)

    print("Creating scene...")
    np.random.seed(42)
    gmap = make_scene(device, 4000)

    print("Generating trajectory...")
    gt_poses = circular_poses(n_frames)

    print("Rendering GT frames...")
    gt_images = [render_at(gmap, p, K_torch, W, H, device) for p in gt_poses]

    print("Running SplatSLAM...")
    import gtsam
    slam = SplatSLAM(
        K=K_np, W=W, H=H, n_pixel_samples=512, device=device,
        mapping_iters=0, photo_sigma=0.5, odom_sigma=0.15, tracking_iters=5,
    )
    slam.gaussian_map = gmap

    est_poses_list = []
    for i in range(n_frames):
        noise = np.eye(4)
        noise[:3, 3] += np.random.randn(3) * 0.08
        init = gt_poses[i] @ noise
        est = slam.add_keyframe(gt_images[i], init_pose=init,
                                prior_sigma=0.5 if i == 0 else None)
        est_poses_list.append(est)
        gt_t = gt_poses[i][:3, 3]
        err = np.linalg.norm(gt_t - est[:3, 3])
        print(f"  Frame {i:2d}: err={err:.3f}m")

    # Loop closure
    rel = np.linalg.inv(gt_poses[0]) @ gt_poses[-1]
    slam.add_loop_closure(0, n_frames - 1, rel, sigma=0.01)
    corrected = slam.get_all_poses()

    print("\nRendering estimated views...")
    est_images = [render_at(gmap, corrected[i], K_torch, W, H, device) for i in range(n_frames)]

    # --- Build video ---
    print("Creating demo video...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output/demo.mp4", fourcc, 4, (W * 2, H))

    for i in range(n_frames):
        gt_frame = to_uint8(gt_images[i])
        est_frame = to_uint8(est_images[i])

        gt_bgr = cv2.cvtColor(gt_frame, cv2.COLOR_RGB2BGR)
        est_bgr = cv2.cvtColor(est_frame, cv2.COLOR_RGB2BGR)

        err = np.linalg.norm(gt_poses[i][:3, 3] - corrected[i][:3, 3])
        add_text(gt_bgr, f"GT Frame {i}", (10, 25), (0, 255, 0))
        add_text(est_bgr, f"Est Frame {i} (err: {err:.2f}m)", (10, 25), (0, 100, 255))

        combined = np.hstack([gt_bgr, est_bgr])
        out.write(combined)

    # Hold last frame
    for _ in range(8):
        out.write(combined)
    out.release()
    print("  Saved output/demo.mp4")

    # --- Comparison grid ---
    print("Creating comparison grid...")
    grid_indices = [0, n_frames//4, n_frames//2, 3*n_frames//4]
    rows = []
    for i in grid_indices:
        gt_u8 = cv2.cvtColor(to_uint8(gt_images[i]), cv2.COLOR_RGB2BGR)
        est_u8 = cv2.cvtColor(to_uint8(est_images[i]), cv2.COLOR_RGB2BGR)
        err = np.linalg.norm(gt_poses[i][:3, 3] - corrected[i][:3, 3])
        add_text(gt_u8, f"GT {i}", (5, 20), (0, 255, 0))
        add_text(est_u8, f"Est {i} ({err:.2f}m)", (5, 20), (0, 100, 255))
        rows.append(np.hstack([gt_u8, est_u8]))
    grid = np.vstack(rows)
    cv2.imwrite("output/comparison_grid.png", grid)
    print("  Saved output/comparison_grid.png")

    # --- Trajectory plot ---
    print("Creating trajectory plot...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    gt_xy = np.array([p[:3, 3] for p in gt_poses])
    est_xy = np.array([corrected[i][:3, 3] for i in range(n_frames)])

    ax.plot(gt_xy[:, 0], gt_xy[:, 2], "g-o", label="Ground Truth", markersize=6, linewidth=2)
    ax.plot(est_xy[:, 0], est_xy[:, 2], "r-x", label="SplatSLAM", markersize=6, linewidth=2)
    for i in range(n_frames):
        ax.plot([gt_xy[i, 0], est_xy[i, 0]], [gt_xy[i, 2], est_xy[i, 2]],
                "k--", alpha=0.3, linewidth=0.5)

    ate = np.mean([np.linalg.norm(gt_poses[i][:3, 3] - corrected[i][:3, 3]) for i in range(n_frames)])
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Z (m)", fontsize=12)
    ax.set_title(f"gtsam-splatfactors: Trajectory (ATE: {ate:.3f}m)", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.savefig("output/trajectory.png", dpi=150, bbox_inches="tight")
    print("  Saved output/trajectory.png")

    print(f"\nATE: {ate:.3f}m")
    print("Done! Check output/ for demo.mp4, comparison_grid.png, trajectory.png")


if __name__ == "__main__":
    main()
