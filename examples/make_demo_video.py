"""Generate the visual demo for the README.

Builds a textured room scene, tracks camera poses with photometric factors,
runs iSAM2 with loop closure, and generates:
  output/demo.mp4            — side-by-side GT vs estimated
  output/trajectory.png      — trajectory + per-frame error
  output/comparison_grid.png — 4 keyframes GT vs estimated

Usage:
    python examples/make_demo_video.py
"""

import numpy as np
import torch
import torch.nn as nn
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


def build_room_scene(device="cuda"):
    """Textured room: wood floor, colored walls, table with objects, painting."""
    np.random.seed(0)
    gs, cs = [], []

    # Wood floor with stripes
    for x in np.linspace(-3, 3, 80):
        for z in np.linspace(1, 10, 120):
            s = 0.4 + 0.3 * np.sin(x * 8) + 0.1 * np.sin(z * 3)
            gs.append([x, -0.8, z]); cs.append([s*0.9, s*0.65, s*0.4])

    # Brick back wall
    for x in np.linspace(-3, 3, 80):
        for y in np.linspace(-0.8, 2.5, 50):
            b = 0.55 + 0.2 * ((int(x*4) + int(y*6)) % 2)
            gs.append([x, y, 10]); cs.append([b*0.85, b*0.45, b*0.35])

    # Blue left wall with stripes
    for y in np.linspace(-0.8, 2.5, 40):
        for z in np.linspace(1, 10, 80):
            s = 0.3 + 0.2 * np.sin(z * 5)
            gs.append([-3, y, z]); cs.append([0.25, 0.3+s*0.3, 0.5+s*0.2])

    # Green right wall with stripes
    for y in np.linspace(-0.8, 2.5, 40):
        for z in np.linspace(1, 10, 80):
            s = 0.3 + 0.2 * np.sin(z * 4 + 1)
            gs.append([3, y, z]); cs.append([0.3+s*0.2, 0.5+s*0.3, 0.25])

    # Table surface
    for x in np.linspace(-0.8, 0.8, 25):
        for z in np.linspace(4, 5.5, 25):
            gs.append([x, 0.2, z]); cs.append([0.45, 0.3, 0.2])

    # Table legs
    for tx, tz in [(-0.7, 4.1), (0.7, 4.1), (-0.7, 5.4), (0.7, 5.4)]:
        for y in np.linspace(-0.8, 0.2, 15):
            gs.append([tx, y, tz]); cs.append([0.35, 0.22, 0.15])

    # Red cup, green box, blue vase on table
    for _ in range(300):
        p = np.array([-0.3, 0.5, 4.5]) + np.random.randn(3)*[0.08, 0.15, 0.08]
        gs.append(p.tolist()); cs.append([0.85, 0.12, 0.1])
    for _ in range(300):
        p = np.array([0.3, 0.4, 4.8]) + np.random.randn(3)*[0.12, 0.1, 0.12]
        gs.append(p.tolist()); cs.append([0.12, 0.7, 0.15])
    for _ in range(250):
        p = np.array([0.0, 0.55, 5.2]) + np.random.randn(3)*[0.06, 0.2, 0.06]
        gs.append(p.tolist()); cs.append([0.15, 0.2, 0.82])

    # Colorful painting on back wall
    for x in np.linspace(-1, 1, 30):
        for y in np.linspace(0.5, 1.8, 20):
            r = 0.5 + 0.5*np.sin(x*5)*np.cos(y*5)
            g = 0.5 + 0.5*np.cos(x*3 + y*4)
            b = 0.5 + 0.3*np.sin(x*7 - y*2)
            gs.append([x, y, 9.9]); cs.append([r, g, b])

    means = torch.tensor(gs, dtype=torch.float32, device=device)
    colors = torch.tensor(np.clip(cs, 0, 1), dtype=torch.float32, device=device)
    n = len(means)
    quats = torch.zeros(n, 4, device=device); quats[:, 0] = 1.0
    scales = torch.full((n, 3), 0.04, device=device)
    opacities = torch.full((n,), 0.97, device=device)

    gmap = GaussianMap(n_gaussians=0, device=device)
    gmap.means = nn.Parameter(means)
    gmap.quats = nn.Parameter(quats)
    gmap.scales = nn.Parameter(scales)
    gmap.opacities = nn.Parameter(opacities)
    gmap.colors = nn.Parameter(colors)
    return gmap


def make_trajectory(n, radius=0.8, cz=5.5):
    """Smooth elliptical trajectory through the room."""
    poses = []
    for i in range(n):
        a = 2 * np.pi * i / n
        x = radius * np.cos(a)
        z = cz + radius * 0.6 * np.sin(a)
        fwd = np.array([0, 0, cz]) - np.array([x, 0.3, z])
        fwd /= np.linalg.norm(fwd)
        right = np.cross([0, 1, 0], fwd); right /= np.linalg.norm(right)
        up = np.cross(fwd, right)
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, up, fwd, [x, 0.3, z]
        poses.append(T)
    return poses


def render_frame(gmap, pose, K, W, H, device):
    vm = torch.inverse(torch.tensor(pose, dtype=torch.float32, device=device))
    with torch.no_grad():
        r, _, _ = render_gaussians(
            gmap.means, gmap.quats, gmap.scales, gmap.opacities, gmap.colors,
            vm, K, W, H)
    return np.clip(r.cpu().numpy(), 0, 1)


def track_frame(gmap, target, init_pose, K, W, H, device, n_pix=1024, sigma=0.3, iters=8):
    key = gtsam.symbol('t', 0)
    p0 = matrix_to_pose3(init_pose)
    g = gtsam.NonlinearFactorGraph()
    v = gtsam.Values()
    g.addPriorPose3(key, p0, gtsam.noiseModel.Isotropic.Sigma(6, 2.0))
    pix = sample_pixel_indices(H, W, n_pix)
    f = GaussianSplatFactor(gmap, target, K, pix, W, H, device)
    g.add(f.as_gtsam_factor(key, gtsam.noiseModel.Isotropic.Sigma(f.n_residuals, sigma)))
    v.insert(key, p0)
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(iters)
    params.setVerbosityLM("SILENT")
    return pose3_to_matrix(gtsam.LevenbergMarquardtOptimizer(g, v, params).optimize().atPose3(key))


def to_u8(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def label(img, text, color=(255, 255, 255)):
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():
    device = "cuda"
    H, W = 300, 400
    K_np = np.array([[200, 0, 200], [0, 200, 150], [0, 0, 1]], dtype=np.float64)
    K_t = torch.tensor(K_np, dtype=torch.float32, device=device)
    n_frames = 24
    os.makedirs("output", exist_ok=True)

    print("=== gtsam-splatfactors Demo ===\n")

    print("1. Building room scene...")
    gmap = build_room_scene(device)
    print(f"   {gmap.n_gaussians} Gaussians\n")

    print("2. Generating trajectory...")
    gt_poses = make_trajectory(n_frames)

    print("3. Rendering GT frames...")
    gt_imgs = [render_frame(gmap, p, K_t, W, H, device) for p in gt_poses]

    print("4. Tracking with photometric factors...")
    np.random.seed(42)
    tracked, err_init, err_track = [], [], []
    for i in range(n_frames):
        noisy = gt_poses[i].copy()
        noisy[:3, 3] += np.random.randn(3) * 0.05
        ei = np.linalg.norm(gt_poses[i][:3, 3] - noisy[:3, 3])
        est = track_frame(gmap, gt_imgs[i], noisy, K_t, W, H, device)
        et = np.linalg.norm(gt_poses[i][:3, 3] - est[:3, 3])
        tracked.append(est)
        err_init.append(ei); err_track.append(et)
        tag = "OK" if et < ei else "!!"
        print(f"   Frame {i:2d}: {ei:.3f}m -> {et:.3f}m  {tag}")

    print(f"\n   ATE: {np.mean(err_init):.3f}m -> {np.mean(err_track):.3f}m")

    print("\n5. iSAM2 + loop closure...")
    isam = gtsam.ISAM2(gtsam.ISAM2Params())
    for i in range(n_frames):
        g = gtsam.NonlinearFactorGraph()
        v = gtsam.Values()
        k = gtsam.symbol('x', i)
        if i == 0:
            g.addPriorPose3(k, matrix_to_pose3(tracked[0]), gtsam.noiseModel.Isotropic.Sigma(6, 0.01))
        else:
            pk = gtsam.symbol('x', i-1)
            odom = matrix_to_pose3(tracked[i-1]).between(matrix_to_pose3(tracked[i]))
            g.add(gtsam.BetweenFactorPose3(pk, k, odom, gtsam.noiseModel.Isotropic.Sigma(6, 0.05)))
        v.insert(k, matrix_to_pose3(tracked[i]))
        isam.update(g, v)

    # Loop closure
    lc = gtsam.NonlinearFactorGraph()
    lc.add(gtsam.BetweenFactorPose3(
        gtsam.symbol('x', 0), gtsam.symbol('x', n_frames-1),
        matrix_to_pose3(gt_poses[0]).between(matrix_to_pose3(gt_poses[-1])),
        gtsam.noiseModel.Isotropic.Sigma(6, 0.02)))
    isam.update(lc, gtsam.Values())
    for _ in range(3): isam.update()
    est = isam.calculateEstimate()
    final = {i: pose3_to_matrix(est.atPose3(gtsam.symbol('x', i))) for i in range(n_frames)}
    err_final = [np.linalg.norm(gt_poses[i][:3, 3] - final[i][:3, 3]) for i in range(n_frames)]
    print(f"   ATE after loop closure: {np.mean(err_final):.3f}m\n")

    print("6. Generating visuals...")
    est_imgs = [render_frame(gmap, final[i], K_t, W, H, device) for i in range(n_frames)]

    # Video
    out = cv2.VideoWriter("output/demo.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 3, (W*2, H))
    for i in range(n_frames):
        gt_bgr = cv2.cvtColor(to_u8(gt_imgs[i]), cv2.COLOR_RGB2BGR)
        est_bgr = cv2.cvtColor(to_u8(est_imgs[i]), cv2.COLOR_RGB2BGR)
        label(gt_bgr, f"Ground Truth - Frame {i}", (0, 255, 0))
        label(est_bgr, f"Estimated ({err_final[i]:.3f}m err)", (0, 180, 255))
        out.write(np.hstack([gt_bgr, est_bgr]))
    for _ in range(6): out.write(np.hstack([gt_bgr, est_bgr]))
    out.release()
    print("   output/demo.mp4")

    # Comparison grid
    idx = [0, n_frames//4, n_frames//2, 3*n_frames//4]
    rows = []
    for i in idx:
        g8 = cv2.cvtColor(to_u8(gt_imgs[i]), cv2.COLOR_RGB2BGR)
        e8 = cv2.cvtColor(to_u8(est_imgs[i]), cv2.COLOR_RGB2BGR)
        label(g8, f"GT Frame {i}", (0, 255, 0))
        label(e8, f"Estimated ({err_final[i]:.3f}m)", (0, 180, 255))
        rows.append(np.hstack([g8, e8]))
    cv2.imwrite("output/comparison_grid.png", np.vstack(rows))
    print("   output/comparison_grid.png")

    # Trajectory + error bar chart
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    gt_xz = np.array([p[:3, 3] for p in gt_poses])
    est_xz = np.array([final[i][:3, 3] for i in range(n_frames)])
    ax1.plot(gt_xz[:, 0], gt_xz[:, 2], "g-o", label="Ground Truth", ms=5, lw=2)
    ax1.plot(est_xz[:, 0], est_xz[:, 2], "r-x", label="SplatSLAM", ms=5, lw=2)
    for i in range(n_frames):
        ax1.plot([gt_xz[i,0], est_xz[i,0]], [gt_xz[i,2], est_xz[i,2]], "k--", alpha=0.3, lw=0.5)
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Z (m)")
    ax1.set_title(f"Trajectory (ATE: {np.mean(err_final):.3f}m)")
    ax1.legend(); ax1.set_aspect("equal"); ax1.grid(alpha=0.3)
    x = range(n_frames)
    ax2.bar(x, err_init, alpha=0.4, color="orange", label="Initial noise")
    ax2.bar(x, err_track, alpha=0.7, color="royalblue", label="After tracking")
    ax2.bar(x, err_final, alpha=0.9, color="green", label="After loop closure")
    ax2.set_xlabel("Frame"); ax2.set_ylabel("Error (m)")
    ax2.set_title("Per-frame Translation Error"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/trajectory.png", dpi=150, bbox_inches="tight")
    print("   output/trajectory.png")

    improved = sum(1 for a, b in zip(err_track, err_init) if a < b)
    print(f"\n=== Results: ATE {np.mean(err_init):.3f}m -> {np.mean(err_track):.3f}m -> {np.mean(err_final):.3f}m | {improved}/{n_frames} improved ===")


if __name__ == "__main__":
    main()
