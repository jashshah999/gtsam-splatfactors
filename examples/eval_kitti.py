"""Evaluate gtsam-splatfactors on KITTI Odometry.

Uses PnP visual odometry + iSAM2 + DINOv2 loop closure.
Demonstrates the system on outdoor automotive data (Frank's request).

Usage:
    python examples/eval_kitti.py --seq 0 --max-frames 200
    python examples/eval_kitti.py --seq 5 --stride 2 --max-frames 500
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gtsam
from gsplat_slam.kitti_loader import KITTIStereoDataset
from gsplat_slam.pose_utils import matrix_to_pose3, pose3_to_matrix


def pnp_odometry(prev_rgb, curr_rgb, prev_depth, K, prev_pose):
    """Compute relative pose via ORB feature matching + PnP."""
    orb = cv2.ORB_create(3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    prev_gray = (prev_rgb * 255).astype(np.uint8)
    curr_gray = (curr_rgb * 255).astype(np.uint8)
    if prev_gray.ndim == 3:
        prev_gray = cv2.cvtColor(prev_gray, cv2.COLOR_RGB2GRAY)
    if curr_gray.ndim == 3:
        curr_gray = cv2.cvtColor(curr_gray, cv2.COLOR_RGB2GRAY)

    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        return prev_pose

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)[:800]

    if len(matches) < 10:
        return prev_pose

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = prev_depth.shape if prev_depth is not None else (prev_rgb.shape[0], prev_rgb.shape[1])

    pts_3d = []
    pts_2d = []
    for m in matches:
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        col, row = int(round(pt1[0])), int(round(pt1[1]))
        if prev_depth is None:
            continue
        if 0 <= row < H and 0 <= col < W:
            d = prev_depth[row, col]
            if 1.0 < d < 50.0:  # cap at 50m for reliable stereo depth
                x = (col - cx) * d / fx
                y = (row - cy) * d / fy
                pt_cam = np.array([x, y, d, 1.0])
                pt_world = (prev_pose @ pt_cam)[:3]
                pts_3d.append(pt_world)
                pts_2d.append(pt2)

    if len(pts_3d) < 8:
        return prev_pose

    pts_3d = np.array(pts_3d, dtype=np.float64)
    pts_2d = np.array(pts_2d, dtype=np.float64)
    cam_mat = K.astype(np.float64)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, cam_mat, None,
        iterationsCount=500, reprojectionError=2.0,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success or inliers is None or len(inliers) < 8:
        return prev_pose

    R, _ = cv2.Rodrigues(rvec)
    T_c2w = np.eye(4)
    T_c2w[:3, :3] = R.T
    T_c2w[:3, 3] = -R.T @ tvec.flatten()
    return T_c2w


def compute_ate(gt_poses, est_poses):
    """Absolute Trajectory Error (translational RMSE)."""
    errors = []
    for g, e in zip(gt_poses, est_poses):
        errors.append(np.linalg.norm(g[:3, 3] - e[:3, 3]))
    return np.sqrt(np.mean(np.array(errors) ** 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--data-root", default="data/kitti")
    parser.add_argument("--output", default="output/kitti")
    parser.add_argument("--loop-closure", action="store_true", default=True)
    parser.add_argument("--lc-threshold", type=float, default=0.85)
    parser.add_argument("--lc-min-gap", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"=== KITTI Odometry: Sequence {args.seq:02d} ===\n")

    dataset = KITTIStereoDataset(
        seq_id=args.seq, data_root=args.data_root,
        stride=args.stride, max_frames=args.max_frames,
    )
    K = dataset.K
    H, W = dataset.H, dataset.W
    print(f"  {len(dataset)} frames, {W}x{H}")
    print(f"  K = diag({K[0,0]:.1f}, {K[1,1]:.1f}), c=({K[0,2]:.1f}, {K[1,2]:.1f})\n")

    # --- Phase 1: Visual Odometry ---
    print("Phase 1: PnP Visual Odometry...")
    gt_poses = []
    vo_poses = []
    prev_frame = None

    for i in range(len(dataset)):
        frame = dataset[i]
        gt_poses.append(frame["pose"])

        if i == 0:
            vo_poses.append(frame["pose"].copy())
            prev_frame = frame
            continue

        est = pnp_odometry(
            prev_frame["rgb"], frame["rgb"],
            prev_frame["depth"], K, vo_poses[-1],
        )
        vo_poses.append(est)
        prev_frame = frame

        if i % 50 == 0:
            err = np.linalg.norm(frame["pose"][:3, 3] - est[:3, 3])
            print(f"  Frame {i:4d}: pos_err={err:.3f}m")

    ate_vo = compute_ate(gt_poses, vo_poses)
    print(f"\n  VO ATE (RMSE): {ate_vo:.4f}m")

    # --- Phase 2: iSAM2 ---
    print("\nPhase 2: iSAM2 optimization...")
    isam_params = gtsam.ISAM2Params()
    isam = gtsam.ISAM2(isam_params)

    for i in range(len(vo_poses)):
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        key = gtsam.symbol('x', i)

        if i == 0:
            noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.01)
            graph.addPriorPose3(key, matrix_to_pose3(vo_poses[0]), noise)
        else:
            prev_key = gtsam.symbol('x', i - 1)
            rel = matrix_to_pose3(vo_poses[i - 1]).between(matrix_to_pose3(vo_poses[i]))
            odom_noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Cauchy.Create(1.0),
                gtsam.noiseModel.Isotropic.Sigma(6, 0.1),
            )
            graph.add(gtsam.BetweenFactorPose3(prev_key, key, rel, odom_noise))

        values.insert(key, matrix_to_pose3(vo_poses[i]))
        isam.update(graph, values)

    estimate = isam.calculateEstimate()
    isam_poses = [pose3_to_matrix(estimate.atPose3(gtsam.symbol('x', i)))
                  for i in range(len(vo_poses))]
    ate_isam = compute_ate(gt_poses, isam_poses)
    print(f"  iSAM2 ATE (RMSE): {ate_isam:.4f}m")

    # --- Phase 3: Loop Closure ---
    n_lc = 0
    if args.loop_closure:
        print("\nPhase 3: DINOv2 Loop Closure...")
        try:
            import torch
            from gsplat_slam.loop_detector import LoopDetector
            HAS_LC = True
        except ImportError:
            HAS_LC = False
            print("  (skipped: torch/DINOv2 not available)")

        if HAS_LC:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            detector = LoopDetector(
                similarity_threshold=args.lc_threshold,
                min_frame_gap=args.lc_min_gap,
                device=device,
            )

            # Detect loops (add_frame builds database, detect queries it)
            print("  Building DINOv2 descriptors and detecting loops...")
            loops = []
            for i in range(len(dataset)):
                frame = dataset[i]
                detector.add_frame(frame["rgb"], i)
                if i >= args.lc_min_gap:
                    matches = detector.detect(frame["rgb"], i)
                    for j, score in matches:
                        loops.append((j, i, score))

            # Keep only top-k highest confidence loops to avoid false positives
            loops.sort(key=lambda x: -x[2])
            max_loops = min(20, len(loops))
            loops = loops[:max_loops]
            print(f"  Detected {len(loops)} loop closure candidates (top-{max_loops})")

            # Add loop closure factors with geometric verification
            graph_lc = gtsam.NonlinearFactorGraph()
            for j, i, score in loops:
                if dataset[j]["depth"] is not None:
                    # Compute relative pose: PnP gives T_cj (pose of camera i in frame j's 3D)
                    rel_pose = pnp_odometry(
                        dataset[j]["rgb"], dataset[i]["rgb"],
                        dataset[j]["depth"], K, np.eye(4),
                    )
                    # Sanity: relative translation should be small for a true loop
                    rel_t = np.linalg.norm(rel_pose[:3, 3])
                    if 0.1 < rel_t < 30.0:
                        key_j = gtsam.symbol('x', j)
                        key_i = gtsam.symbol('x', i)
                        between = matrix_to_pose3(rel_pose)
                        lc_noise = gtsam.noiseModel.Robust.Create(
                            gtsam.noiseModel.mEstimator.Cauchy.Create(1.0),
                            gtsam.noiseModel.Isotropic.Sigma(6, 0.5),
                        )
                        graph_lc.add(gtsam.BetweenFactorPose3(key_j, key_i, between, lc_noise))
                        n_lc += 1

            if n_lc > 0:
                isam.update(graph_lc, gtsam.Values())
                estimate = isam.calculateEstimate()
                final_poses = [pose3_to_matrix(estimate.atPose3(gtsam.symbol('x', i)))
                               for i in range(len(vo_poses))]
                ate_lc = compute_ate(gt_poses, final_poses)
                print(f"  Added {n_lc} loop closure factors")
                print(f"  iSAM2+LC ATE (RMSE): {ate_lc:.4f}m")
            else:
                final_poses = isam_poses
                ate_lc = ate_isam
    else:
        final_poses = isam_poses
        ate_lc = ate_isam

    # --- Results ---
    print(f"\n{'='*50}")
    print(f"KITTI Sequence {args.seq:02d} Results:")
    print(f"  VO ATE:       {ate_vo:.4f}m")
    print(f"  iSAM2 ATE:   {ate_isam:.4f}m")
    if n_lc > 0:
        print(f"  iSAM2+LC ATE: {ate_lc:.4f}m  ({n_lc} loop closures)")
        improvement = (ate_vo - ate_lc) / ate_vo * 100
        print(f"  Improvement:  {improvement:.1f}%")
    print(f"{'='*50}")

    # --- Plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_t = np.array([p[:3, 3] for p in gt_poses])
    vo_t = np.array([p[:3, 3] for p in vo_poses])
    final_t = np.array([p[:3, 3] for p in final_poses])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bird's eye view (X-Z plane)
    ax = axes[0]
    ax.plot(gt_t[:, 0], gt_t[:, 2], 'g-', label='Ground Truth', lw=2)
    ax.plot(vo_t[:, 0], vo_t[:, 2], 'r-', label=f'VO (ATE={ate_vo:.2f}m)', lw=1, alpha=0.7)
    ax.plot(final_t[:, 0], final_t[:, 2], 'b-', label=f'iSAM2 (ATE={ate_lc:.2f}m)', lw=1.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(f"KITTI {args.seq:02d} — Bird's Eye View")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    # Per-frame error
    ax = axes[1]
    errs_vo = [np.linalg.norm(g[:3, 3] - e[:3, 3]) for g, e in zip(gt_poses, vo_poses)]
    errs_final = [np.linalg.norm(g[:3, 3] - e[:3, 3]) for g, e in zip(gt_poses, final_poses)]
    ax.plot(errs_vo, 'r-', label='VO', alpha=0.7)
    ax.plot(errs_final, 'b-', label='iSAM2+LC')
    ax.set_xlabel("Frame")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Per-frame ATE")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(args.output, f"kitti_{args.seq:02d}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nTrajectory plot saved: {out_path}")

    # Save trajectory in KITTI format
    traj_path = os.path.join(args.output, f"kitti_{args.seq:02d}_est.txt")
    with open(traj_path, "w") as f:
        for pose in final_poses:
            row = pose[:3, :].flatten()
            f.write(" ".join(f"{v:.6e}" for v in row) + "\n")
    print(f"Trajectory saved: {traj_path}")


if __name__ == "__main__":
    main()
