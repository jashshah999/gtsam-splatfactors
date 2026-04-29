"""Evaluate gtsam-splatfactors on TUM-RGBD.

Phase 1: Build + train Gaussian map from first N frames (GT poses)
Phase 2: Track remaining frames with photometric factors (LM)
Phase 3: iSAM2 global optimization with loop closure

Usage:
    python examples/eval_tum.py
    python examples/eval_tum.py --seq fr1/xyz --train-iters 500
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gtsam
from gsplat_slam.tum_loader import TUMDataset
from gsplat_slam.mapper import GaussianMapper
from gsplat_slam.renderer import sample_pixel_indices
from gsplat_slam.factor import GaussianSplatFactor
from gsplat_slam.pose_utils import matrix_to_pose3, pose3_to_matrix


def track_frame(mapper, target_rgb, init_pose, K_torch, W, H, device,
                n_pixels=1024, sigma=0.5, iters=8):
    """Track one frame via LM optimization with photometric factor."""
    # Build a temporary GaussianMap-like object from mapper params
    from gsplat_slam.map import GaussianMap
    import torch.nn as nn
    gmap = GaussianMap(n_gaussians=0, device=device)
    gmap.means = nn.Parameter(mapper.params["means"].data.clone())
    gmap.quats = nn.Parameter(mapper.params["quats"].data.clone())
    gmap.scales = nn.Parameter(mapper.params["scales"].data.clone())
    gmap.opacities = nn.Parameter(mapper.params["opacities"].data.clone())
    gmap.colors = nn.Parameter(mapper.params["colors"].data.clone())

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="fr1/desk")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--init-frames", type=int, default=15)
    parser.add_argument("--train-iters", type=int, default=500)
    parser.add_argument("--n-pixels", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="output/tum")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = args.device

    print(f"=== TUM-RGBD: {args.seq} ===\n")

    dataset = TUMDataset(args.seq, stride=args.stride, max_frames=args.max_frames)
    K = dataset.K
    K_torch = torch.tensor(K, dtype=torch.float32, device=device)
    H, W = dataset.H, dataset.W
    print(f"  {len(dataset)} frames, {W}x{H}\n")

    # === Phase 1: Build and train map ===
    print(f"Phase 1: Building map from {args.init_frames} frames...")
    mapper = GaussianMapper(device=device, refine_start=100, refine_every=50, reset_every=500)

    init_frames = []
    for i in range(min(args.init_frames, len(dataset))):
        frame = dataset[i]
        mapper.init_from_rgbd(frame["rgb"], frame["depth"], frame["pose"], K, stride=4)
        init_frames.append(frame)

    print(f"  {mapper.n_gaussians} initial Gaussians")
    print(f"  Training for {args.train_iters} iterations...")
    mapper.train_on_frames(init_frames, K_torch, W, H,
                           n_iters=args.train_iters, log_every=100)

    # Verify: render a trained view and save
    test_render = mapper.render(init_frames[0]["pose"], K_torch, W, H)
    cv2.imwrite(os.path.join(args.output, "trained_render.png"),
                cv2.cvtColor((test_render * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    gt_img = (init_frames[0]["rgb"] * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output, "gt_frame0.png"),
                cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR))
    print(f"  Saved trained render vs GT to {args.output}/\n")

    # === Phase 2: Track ===
    print("Phase 2: Tracking...")
    gt_poses, tracked_poses, timestamps = [], [], []

    for i in range(len(dataset)):
        frame = dataset[i]
        gt_poses.append(frame["pose"])
        timestamps.append(frame["timestamp"])

        if i < args.init_frames:
            tracked_poses.append(frame["pose"])
            continue

        init_pose = tracked_poses[-1].copy()
        t0 = time.time()
        est = track_frame(mapper, frame["rgb"], init_pose, K_torch, W, H, device,
                          n_pixels=args.n_pixels, sigma=0.5, iters=8)
        dt = time.time() - t0
        err = np.linalg.norm(frame["pose"][:3, 3] - est[:3, 3])
        tracked_poses.append(est)
        print(f"  {i:3d}: err={err:.4f}m  ({dt:.1f}s)")

        # Incremental mapping every 5 frames
        if i % 5 == 0:
            kf = {"rgb": frame["rgb"], "pose": est}
            mapper.train_on_frames([kf], K_torch, W, H, n_iters=30, log_every=100)

    ate_track = np.mean([np.linalg.norm(g[:3,3] - e[:3,3])
                         for g, e in zip(gt_poses, tracked_poses)])
    print(f"\n  ATE (tracked): {ate_track:.4f}m")

    # === Phase 3: iSAM2 ===
    print("\nPhase 3: iSAM2...")
    isam = gtsam.ISAM2(gtsam.ISAM2Params())
    for i in range(len(tracked_poses)):
        g = gtsam.NonlinearFactorGraph()
        v = gtsam.Values()
        k = gtsam.symbol('x', i)
        if i == 0:
            g.addPriorPose3(k, matrix_to_pose3(tracked_poses[0]),
                           gtsam.noiseModel.Isotropic.Sigma(6, 0.01))
        else:
            odom = matrix_to_pose3(tracked_poses[i-1]).between(matrix_to_pose3(tracked_poses[i]))
            g.add(gtsam.BetweenFactorPose3(gtsam.symbol('x', i-1), k, odom,
                  gtsam.noiseModel.Isotropic.Sigma(6, 0.05)))
        v.insert(k, matrix_to_pose3(tracked_poses[i]))
        isam.update(g, v)

    estimate = isam.calculateEstimate()
    final = [pose3_to_matrix(estimate.atPose3(gtsam.symbol('x', i)))
             for i in range(len(tracked_poses))]

    ate_final = np.mean([np.linalg.norm(g[:3,3] - e[:3,3])
                         for g, e in zip(gt_poses, final)])
    print(f"  ATE (iSAM2): {ate_final:.4f}m")

    # === Save results ===
    print(f"\n=== Results: {args.seq} ===")
    print(f"  ATE tracked: {ate_track:.4f}m")
    print(f"  ATE iSAM2:   {ate_final:.4f}m")

    # Trajectory plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    gt_t = np.array([p[:3, 3] for p in gt_poses])
    est_t = np.array([p[:3, 3] for p in final])

    ax1.plot(gt_t[:, 0], gt_t[:, 2], "g-", label="GT", lw=2)
    ax1.plot(est_t[:, 0], est_t[:, 2], "r-", label="SplatSLAM", lw=1.5)
    ax1.set_xlabel("X"); ax1.set_ylabel("Z")
    ax1.set_title(f"{args.seq} — ATE: {ate_final:.4f}m"); ax1.legend()
    ax1.set_aspect("equal"); ax1.grid(alpha=0.3)

    errs = [np.linalg.norm(g[:3,3] - e[:3,3]) for g, e in zip(gt_poses, final)]
    ax2.plot(errs); ax2.set_xlabel("Frame"); ax2.set_ylabel("Error (m)")
    ax2.set_title("Per-frame ATE"); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output, f"tum_{args.seq.replace('/', '_')}.png"),
                dpi=150, bbox_inches="tight")

    # Save trajectory
    from scipy.spatial.transform import Rotation
    traj_path = os.path.join(args.output, f"est_{args.seq.replace('/', '_')}.txt")
    with open(traj_path, "w") as f:
        for i in range(len(final)):
            t = final[i][:3, 3]
            q = Rotation.from_matrix(final[i][:3, :3]).as_quat()
            f.write(f"{timestamps[i]:.6f} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")

    # Comparison renders
    for i in [0, len(dataset)//4, len(dataset)//2, 3*len(dataset)//4]:
        if i < len(final):
            r = mapper.render(final[i], K_torch, W, H)
            cv2.imwrite(os.path.join(args.output, f"render_{i:03d}.png"),
                        cv2.cvtColor((r*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    print(f"\nOutputs saved to {args.output}/")


if __name__ == "__main__":
    main()
