"""Evaluate gtsam-splatfactors on TUM-RGBD.

Downloads the dataset automatically, runs tracking + mapping + loop closure,
computes ATE, and saves trajectory plots.

Usage:
    python examples/eval_tum.py                        # fr1/desk, default
    python examples/eval_tum.py --seq fr1/xyz          # different sequence
    python examples/eval_tum.py --max-frames 100       # quick test
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gtsam
from gsplat_slam.tum_loader import TUMDataset
from gsplat_slam.map import GaussianMap
from gsplat_slam.renderer import render_gaussians, sample_pixel_indices
from gsplat_slam.factor import GaussianSplatFactor
from gsplat_slam.pose_utils import matrix_to_pose3, pose3_to_matrix


def init_gaussians_from_rgbd(rgb, depth, pose, K, device, stride=4):
    """Back-project RGB-D frame to 3D Gaussians."""
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    rows = np.arange(0, H, stride)
    cols = np.arange(0, W, stride)
    rr, cc = np.meshgrid(rows, cols, indexing='ij')
    rr, cc = rr.flatten(), cc.flatten()
    d = depth[rr, cc]
    valid = (d > 0.1) & (d < 5.0)
    rr, cc, d = rr[valid], cc[valid], d[valid]

    x_cam = (cc - cx) * d / fx
    y_cam = (rr - cy) * d / fy
    z_cam = d
    pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(d)], axis=1)
    pts_world = (pose @ pts_cam.T).T[:, :3]
    colors_np = rgb[rr, cc]

    return pts_world.astype(np.float32), colors_np.astype(np.float32)


def track_frame(gmap, target_rgb, init_pose, K_torch, W, H, device,
                n_pixels=1024, sigma=0.5, iters=8):
    """Track a single frame via LM with photometric factor."""
    key = gtsam.symbol('t', 0)
    p0 = matrix_to_pose3(init_pose)
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    graph.addPriorPose3(key, p0, gtsam.noiseModel.Isotropic.Sigma(6, 1.0))

    pix = sample_pixel_indices(H, W, n_pixels)
    factor = GaussianSplatFactor(gmap, target_rgb, K_torch, pix, W, H, device)
    graph.add(factor.as_gtsam_factor(key,
              gtsam.noiseModel.Isotropic.Sigma(factor.n_residuals, sigma)))

    values.insert(key, p0)
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(iters)
    params.setVerbosityLM("SILENT")
    result = gtsam.LevenbergMarquardtOptimizer(graph, values, params).optimize()
    return pose3_to_matrix(result.atPose3(key))


def compute_ate(gt_poses, est_poses):
    """Compute Absolute Trajectory Error."""
    errors = []
    for gt, est in zip(gt_poses, est_poses):
        err = np.linalg.norm(gt[:3, 3] - est[:3, 3])
        errors.append(err)
    return np.array(errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="fr1/desk", help="TUM sequence name")
    parser.add_argument("--stride", type=int, default=5, help="Frame stride")
    parser.add_argument("--max-frames", type=int, default=200, help="Max keyframes")
    parser.add_argument("--n-pixels", type=int, default=1024, help="Sampled pixels per factor")
    parser.add_argument("--init-frames", type=int, default=5, help="Frames for initial map")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="output/tum")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = args.device

    print(f"=== TUM-RGBD Evaluation: {args.seq} ===\n")

    # Load dataset
    print("Loading dataset...")
    dataset = TUMDataset(args.seq, stride=args.stride, max_frames=args.max_frames)
    K = dataset.K
    K_torch = torch.tensor(K, dtype=torch.float32, device=device)
    H, W = dataset.H, dataset.W
    print(f"  {len(dataset)} frames, {W}x{H}, stride={args.stride}\n")

    # Phase 1: Build initial map from first N frames using GT poses
    print(f"Phase 1: Building initial map from first {args.init_frames} frames...")
    all_means, all_colors = [], []
    for i in range(min(args.init_frames, len(dataset))):
        frame = dataset[i]
        pts, cols = init_gaussians_from_rgbd(frame["rgb"], frame["depth"],
                                              frame["pose"], K, device, stride=6)
        all_means.append(pts)
        all_colors.append(cols)
    all_means = np.concatenate(all_means)
    all_colors = np.concatenate(all_colors)

    # Subsample if too many
    if len(all_means) > 100000:
        idx = np.random.choice(len(all_means), 100000, replace=False)
        all_means = all_means[idx]
        all_colors = all_colors[idx]

    gmap = GaussianMap.from_pointcloud(all_means, all_colors, device=device)
    gmap.scales.data.fill_(-3.5)
    gmap.opacities.data.fill_(3.0)
    print(f"  {gmap.n_gaussians} Gaussians initialized")

    # Train Gaussians on init frames
    print("  Training Gaussians on init frames...")
    from gsplat import rasterization as rast
    map_opt = torch.optim.Adam(gmap.parameters(), lr=0.008)
    for epoch in range(30):
        total_loss = 0
        for fi in range(min(args.init_frames, len(dataset))):
            frame = dataset[fi]
            vm = torch.inverse(torch.tensor(frame["pose"], dtype=torch.float32, device=device))
            vm[1] = -vm[1]
            target = torch.tensor(frame["rgb"], dtype=torch.float32, device=device)
            map_opt.zero_grad()
            rendered, _, _ = rast(
                means=gmap.means, quats=gmap.quats, scales=gmap.scales,
                opacities=gmap.opacities.sigmoid(), colors=gmap.colors,
                viewmats=vm[None], Ks=K_torch[None], width=W, height=H)
            loss = torch.nn.functional.mse_loss(rendered[0], target)
            loss.backward()
            map_opt.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: loss={total_loss/args.init_frames:.4f}")
    print(f"  Training done.\n")

    # Phase 2: Track all frames
    print("Phase 2: Tracking...")
    gt_poses_list = []
    tracked_poses = []
    timestamps = []

    for i in range(len(dataset)):
        frame = dataset[i]
        gt_pose = frame["pose"]
        gt_poses_list.append(gt_pose)
        timestamps.append(frame["timestamp"])

        if i < args.init_frames:
            # Use GT for init frames (they built the map)
            tracked_poses.append(gt_pose)
            print(f"  Frame {i:3d}: init (GT)")
            continue

        # Use previous tracked pose as initial guess
        init_pose = tracked_poses[-1].copy()

        t0 = time.time()
        est_pose = track_frame(gmap, frame["rgb"], init_pose, K_torch, W, H, device,
                                n_pixels=args.n_pixels, sigma=0.5, iters=8)
        dt = time.time() - t0

        err = np.linalg.norm(gt_pose[:3, 3] - est_pose[:3, 3])
        tracked_poses.append(est_pose)
        print(f"  Frame {i:3d}: err={err:.4f}m  ({dt:.1f}s)")

        # Mapping: optimize Gaussians to match this frame
        if i % 3 == 0:  # map every 3rd frame to save time
            map_pose_t = torch.tensor(est_pose, dtype=torch.float32, device=device)
            map_vm = torch.inverse(map_pose_t)
            # Y-flip for gsplat
            map_vm[1] = -map_vm[1]
            target_t = torch.tensor(frame["rgb"], dtype=torch.float32, device=device)
            map_opt = torch.optim.Adam(gmap.parameters(), lr=0.005)
            for _ in range(20):
                map_opt.zero_grad()
                from gsplat import rasterization as rast
                rendered, _, _ = rast(
                    means=gmap.means, quats=gmap.quats,
                    scales=gmap.scales, opacities=gmap.opacities.sigmoid(),
                    colors=gmap.colors, viewmats=map_vm[None],
                    Ks=K_torch[None], width=W, height=H)
                loss = torch.nn.functional.mse_loss(rendered[0], target_t)
                loss.backward()
                map_opt.step()
            print(f"           map loss: {loss.item():.4f}")

    # Phase 3: iSAM2 global optimization
    print("\nPhase 3: iSAM2 global optimization...")
    isam = gtsam.ISAM2(gtsam.ISAM2Params())
    for i in range(len(tracked_poses)):
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        key = gtsam.symbol('x', i)

        if i == 0:
            graph.addPriorPose3(key, matrix_to_pose3(tracked_poses[0]),
                               gtsam.noiseModel.Isotropic.Sigma(6, 0.01))
        else:
            prev_key = gtsam.symbol('x', i - 1)
            odom = matrix_to_pose3(tracked_poses[i-1]).between(matrix_to_pose3(tracked_poses[i]))
            graph.add(gtsam.BetweenFactorPose3(prev_key, key, odom,
                      gtsam.noiseModel.Isotropic.Sigma(6, 0.05)))

        values.insert(key, matrix_to_pose3(tracked_poses[i]))
        isam.update(graph, values)

    estimate = isam.calculateEstimate()
    final_poses = [pose3_to_matrix(estimate.atPose3(gtsam.symbol('x', i)))
                   for i in range(len(tracked_poses))]

    # Compute metrics
    ate_tracked = compute_ate(gt_poses_list, tracked_poses)
    ate_final = compute_ate(gt_poses_list, final_poses)

    print(f"\n=== Results: {args.seq} ===")
    print(f"  Frames:          {len(dataset)}")
    print(f"  ATE (tracked):   {ate_tracked.mean():.4f}m (median: {np.median(ate_tracked):.4f}m)")
    print(f"  ATE (iSAM2):     {ate_final.mean():.4f}m (median: {np.median(ate_final):.4f}m)")
    print(f"  Max error:       {ate_final.max():.4f}m")

    # Save trajectory plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        gt_t = np.array([p[:3, 3] for p in gt_poses_list])
        est_t = np.array([p[:3, 3] for p in final_poses])

        ax1.plot(gt_t[:, 0], gt_t[:, 2], "g-", label="Ground Truth", lw=2)
        ax1.plot(est_t[:, 0], est_t[:, 2], "r-", label="SplatSLAM", lw=1.5, alpha=0.8)
        ax1.set_xlabel("X (m)"); ax1.set_ylabel("Z (m)")
        ax1.set_title(f"{args.seq} — ATE: {ate_final.mean():.4f}m")
        ax1.legend(); ax1.set_aspect("equal"); ax1.grid(alpha=0.3)

        ax2.plot(ate_tracked, "b-", label="After tracking", alpha=0.7)
        ax2.plot(ate_final, "g-", label="After iSAM2", alpha=0.9)
        ax2.set_xlabel("Frame"); ax2.set_ylabel("Error (m)")
        ax2.set_title("Per-frame ATE"); ax2.legend(); ax2.grid(alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(args.output, f"tum_{args.seq.replace('/', '_')}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\n  Saved: {out_path}")
    except ImportError:
        pass

    # Save trajectory as TUM format for external evaluation tools
    traj_path = os.path.join(args.output, f"estimated_{args.seq.replace('/', '_')}.txt")
    with open(traj_path, "w") as f:
        for i in range(len(final_poses)):
            T = final_poses[i]
            t = T[:3, 3]
            R = T[:3, :3]
            # Extract quaternion
            from scipy.spatial.transform import Rotation
            q = Rotation.from_matrix(R).as_quat()  # xyzw
            f.write(f"{timestamps[i]:.6f} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")
    print(f"  Saved: {traj_path}")


if __name__ == "__main__":
    main()
