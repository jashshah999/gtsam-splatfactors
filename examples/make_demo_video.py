"""Generate a visual demo with a structured scene that actually tracks.

Produces:
  output/demo.mp4            — side-by-side GT vs estimated rendering
  output/trajectory.png      — bird's-eye trajectory plot
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

from gsplat import rasterization
from gsplat_slam.map import GaussianMap
from gsplat_slam.renderer import render_gaussians, sample_pixel_indices
from gsplat_slam.factor import GaussianSplatFactor
from gsplat_slam.pose_utils import matrix_to_pose3, pose3_to_matrix

import gtsam


def build_structured_scene(device="cuda"):
    """Build a scene with real visual features: checkerboard + colored objects."""
    torch.manual_seed(42)
    np.random.seed(42)

    gaussians = []
    colors_list = []

    # Checkerboard floor at y = -0.5
    for x in np.linspace(-2.5, 2.5, 50):
        for z in np.linspace(2, 8, 60):
            ix, iz = int(x * 5), int(z * 5)
            c = 0.85 if (ix + iz) % 2 == 0 else 0.15
            gaussians.append([x, -0.5, z])
            colors_list.append([c, c, c])

    # Back wall with gradient at z=8.5
    for x in np.linspace(-2.5, 2.5, 50):
        for y in np.linspace(-0.5, 2, 25):
            r = 0.3 + 0.4 * (x + 2.5) / 5.0
            g = 0.3 + 0.4 * (y + 0.5) / 2.5
            b = 0.5
            gaussians.append([x, y, 8.5])
            colors_list.append([r, g, b])

    # Colored landmark objects
    landmarks = [
        (-1.2, 0.3, 4.0, [0.9, 0.1, 0.1]),   # red
        (1.2, 0.3, 4.0, [0.1, 0.8, 0.1]),    # green
        (0.0, 0.5, 6.0, [0.1, 0.1, 0.9]),    # blue
        (-1.5, 0.0, 7.0, [0.9, 0.9, 0.1]),   # yellow
        (1.5, 0.0, 7.0, [0.9, 0.1, 0.9]),    # magenta
        (0.0, 0.8, 3.5, [0.1, 0.9, 0.9]),    # cyan
    ]
    for cx, cy, cz, col in landmarks:
        for _ in range(250):
            p = np.array([cx, cy, cz]) + np.random.randn(3) * 0.12
            jitter = np.random.randn(3) * 0.03
            gaussians.append(p.tolist())
            colors_list.append(np.clip(np.array(col) + jitter, 0, 1).tolist())

    means = torch.tensor(gaussians, dtype=torch.float32, device=device)
    colors = torch.tensor(np.clip(colors_list, 0, 1), dtype=torch.float32, device=device)
    n = len(means)
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0
    scales = torch.full((n, 3), 0.06, device=device)
    opacities = torch.full((n,), 0.95, device=device)

    gmap = GaussianMap(n_gaussians=0, device=device)
    from torch import nn
    gmap.means = nn.Parameter(means)
    gmap.quats = nn.Parameter(quats)
    gmap.scales = nn.Parameter(scales)
    gmap.opacities = nn.Parameter(opacities)
    gmap.colors = nn.Parameter(colors)

    return gmap


def make_trajectory(n_frames=20, radius=1.0, center_z=5.0):
    """Small circular trajectory looking inward — stays close to scene."""
    poses = []
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        x = radius * np.cos(angle)
        z = center_z + radius * np.sin(angle) * 0.5  # elliptical, less depth variation

        forward = np.array([0, 0, center_z]) - np.array([x, 0.3, z])
        forward /= np.linalg.norm(forward)
        right = np.cross([0, 1, 0], forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)

        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2] = right, up, forward
        T[:3, 3] = [x, 0.3, z]
        poses.append(T)
    return poses


def render_frame(gmap, pose, K, W, H, device):
    T = torch.tensor(pose, dtype=torch.float32, device=device)
    vm = torch.inverse(T)
    with torch.no_grad():
        r, _, _ = render_gaussians(
            gmap.means, gmap.quats, gmap.scales, gmap.opacities, gmap.colors,
            vm, K, W, H)
    return np.clip(r.cpu().numpy(), 0, 1)


def track_single_frame(gmap, target_image, init_pose, K_torch, W, H, device,
                        n_pixels=1024, sigma=0.3, n_iters=10):
    """Track a single frame with LM optimization against photometric factor."""
    key = gtsam.symbol('t', 0)
    pose_init = matrix_to_pose3(init_pose)

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # Weak prior
    prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 2.0)
    graph.addPriorPose3(key, pose_init, prior_noise)

    # Photometric factor
    pixel_idx = sample_pixel_indices(H, W, n_pixels)
    factor = GaussianSplatFactor(
        gaussian_map=gmap, target_image=target_image,
        K=K_torch, pixel_indices=pixel_idx, W=W, H=H, device=device,
    )
    photo_noise = gtsam.noiseModel.Isotropic.Sigma(factor.n_residuals, sigma)
    graph.add(factor.as_gtsam_factor(key, photo_noise))

    initial.insert(key, pose_init)

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(n_iters)
    params.setVerbosityLM("SILENT")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    return pose3_to_matrix(result.atPose3(key))


def to_uint8(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def add_label(img, text, color=(255, 255, 255)):
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    return img


def main():
    device = "cuda"
    H, W = 240, 320
    K_np = np.array([[160, 0, 160], [0, 160, 120], [0, 0, 1]], dtype=np.float64)
    K_torch = torch.tensor(K_np, dtype=torch.float32, device=device)
    n_frames = 20
    os.makedirs("output", exist_ok=True)

    print("=== gtsam-splatfactors Demo ===\n")

    print("1. Building structured scene...")
    gmap = build_structured_scene(device)
    print(f"   {gmap.n_gaussians} Gaussians\n")

    print("2. Generating trajectory...")
    gt_poses = make_trajectory(n_frames, radius=1.0, center_z=5.0)

    print("3. Rendering ground truth frames...")
    gt_images = []
    for p in gt_poses:
        gt_images.append(render_frame(gmap, p, K_torch, W, H, device))

    # Check renders look good
    nz = sum((img.sum(-1) > 0).sum() for img in gt_images)
    print(f"   Avg non-zero pixels: {nz // n_frames}/{H*W}\n")

    print("4. Tracking with photometric factors (LM optimization)...")
    np.random.seed(123)
    tracked_poses = []
    errors_before = []
    errors_after = []

    for i in range(n_frames):
        # Add noise: small rotation + translation perturbation
        noise_t = np.random.randn(3) * 0.05  # 5cm noise
        noise_pose = gt_poses[i].copy()
        noise_pose[:3, 3] += noise_t
        err_before = np.linalg.norm(gt_poses[i][:3, 3] - noise_pose[:3, 3])
        errors_before.append(err_before)

        est_pose = track_single_frame(
            gmap, gt_images[i], noise_pose, K_torch, W, H, device,
            n_pixels=1024, sigma=0.3, n_iters=8,
        )
        tracked_poses.append(est_pose)

        err_after = np.linalg.norm(gt_poses[i][:3, 3] - est_pose[:3, 3])
        errors_after.append(err_after)
        improved = "improved" if err_after < err_before else "worse"
        print(f"   Frame {i:2d}: {err_before:.3f}m -> {err_after:.3f}m ({improved})")

    ate_before = np.mean(errors_before)
    ate_after = np.mean(errors_after)
    print(f"\n   ATE: {ate_before:.3f}m (init) -> {ate_after:.3f}m (tracked)")

    print("\n5. Building iSAM2 graph with odometry + loop closure...")
    isam_params = gtsam.ISAM2Params()
    isam = gtsam.ISAM2(isam_params)

    for i in range(n_frames):
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        key = gtsam.symbol('x', i)

        if i == 0:
            noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.01)
            graph.addPriorPose3(key, matrix_to_pose3(tracked_poses[i]), noise)
        else:
            prev_key = gtsam.symbol('x', i - 1)
            p_prev = matrix_to_pose3(tracked_poses[i - 1])
            p_curr = matrix_to_pose3(tracked_poses[i])
            odom = p_prev.between(p_curr)
            noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.05)
            graph.add(gtsam.BetweenFactorPose3(prev_key, key, odom, noise))

        initial.insert(key, matrix_to_pose3(tracked_poses[i]))
        isam.update(graph, initial)

    # Loop closure
    key_first = gtsam.symbol('x', 0)
    key_last = gtsam.symbol('x', n_frames - 1)
    p0 = matrix_to_pose3(gt_poses[0])
    pN = matrix_to_pose3(gt_poses[-1])
    lc_graph = gtsam.NonlinearFactorGraph()
    lc_graph.add(gtsam.BetweenFactorPose3(
        key_first, key_last, p0.between(pN),
        gtsam.noiseModel.Isotropic.Sigma(6, 0.02)))
    isam.update(lc_graph, gtsam.Values())
    for _ in range(3):
        isam.update()

    estimate = isam.calculateEstimate()
    final_poses = {}
    for i in range(n_frames):
        final_poses[i] = pose3_to_matrix(estimate.atPose3(gtsam.symbol('x', i)))

    errors_final = [np.linalg.norm(gt_poses[i][:3, 3] - final_poses[i][:3, 3]) for i in range(n_frames)]
    ate_final = np.mean(errors_final)
    print(f"   ATE after loop closure: {ate_final:.3f}m\n")

    print("6. Rendering estimated views + generating visuals...")
    est_images = [render_frame(gmap, final_poses[i], K_torch, W, H, device) for i in range(n_frames)]

    # --- Video ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output/demo.mp4", fourcc, 4, (W * 2, H))
    for i in range(n_frames):
        gt_bgr = cv2.cvtColor(to_uint8(gt_images[i]), cv2.COLOR_RGB2BGR)
        est_bgr = cv2.cvtColor(to_uint8(est_images[i]), cv2.COLOR_RGB2BGR)
        add_label(gt_bgr, f"GT {i}", (0, 255, 0))
        err = errors_final[i]
        add_label(est_bgr, f"Est {i} ({err:.2f}m)", (0, 150, 255))
        out.write(np.hstack([gt_bgr, est_bgr]))
    for _ in range(8):
        out.write(np.hstack([gt_bgr, est_bgr]))
    out.release()
    print("   Saved output/demo.mp4")

    # --- Comparison grid ---
    indices = [0, n_frames // 4, n_frames // 2, 3 * n_frames // 4]
    rows = []
    for i in indices:
        gt_u8 = cv2.cvtColor(to_uint8(gt_images[i]), cv2.COLOR_RGB2BGR)
        est_u8 = cv2.cvtColor(to_uint8(est_images[i]), cv2.COLOR_RGB2BGR)
        add_label(gt_u8, f"GT {i}", (0, 255, 0))
        add_label(est_u8, f"Est {i} ({errors_final[i]:.2f}m)", (0, 150, 255))
        rows.append(np.hstack([gt_u8, est_u8]))
    cv2.imwrite("output/comparison_grid.png", np.vstack(rows))
    print("   Saved output/comparison_grid.png")

    # --- Trajectory plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bird's eye
    ax = axes[0]
    gt_xz = np.array([p[:3, 3] for p in gt_poses])
    est_xz = np.array([final_poses[i][:3, 3] for i in range(n_frames)])
    ax.plot(gt_xz[:, 0], gt_xz[:, 2], "g-o", label="Ground Truth", markersize=5, linewidth=2)
    ax.plot(est_xz[:, 0], est_xz[:, 2], "r-x", label="SplatSLAM", markersize=5, linewidth=2)
    for i in range(n_frames):
        ax.plot([gt_xz[i, 0], est_xz[i, 0]], [gt_xz[i, 2], est_xz[i, 2]], "k--", alpha=0.3, lw=0.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"Trajectory (ATE: {ate_final:.3f}m)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Error over time
    ax2 = axes[1]
    ax2.bar(range(n_frames), errors_before, alpha=0.4, color="orange", label="Before tracking")
    ax2.bar(range(n_frames), errors_after, alpha=0.6, color="blue", label="After tracking")
    ax2.bar(range(n_frames), errors_final, alpha=0.8, color="green", label="After loop closure")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Translation Error (m)")
    ax2.set_title("Per-frame Error")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/trajectory.png", dpi=150, bbox_inches="tight")
    print("   Saved output/trajectory.png")

    print(f"\n=== Results ===")
    print(f"  Init ATE:          {ate_before:.3f}m")
    print(f"  After tracking:    {ate_after:.3f}m")
    print(f"  After loop close:  {ate_final:.3f}m")
    print(f"  Frames improved:   {sum(a < b for a, b in zip(errors_after, errors_before))}/{n_frames}")
    print(f"\nDone! Check output/")


if __name__ == "__main__":
    main()
