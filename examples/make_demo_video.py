"""Generate the demo GIF and visuals for the README.

Produces:
  assets/demo.gif            — animated GIF: camera sweep through room
  assets/tracking.gif        — animated GIF: noisy init vs tracked overlay
  assets/trajectory.png      — trajectory plot + error bars

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


def build_room(device="cuda"):
    np.random.seed(0)
    gs, cs = [], []

    for x in np.linspace(-3, 3, 80):
        for z in np.linspace(1, 10, 120):
            s = 0.4 + 0.3*np.sin(x*8) + 0.1*np.sin(z*3)
            gs.append([x, -0.8, z]); cs.append([s*0.9, s*0.65, s*0.4])
    for x in np.linspace(-3, 3, 80):
        for y in np.linspace(-0.8, 2.5, 50):
            b = 0.55 + 0.2*((int(x*4)+int(y*6))%2)
            gs.append([x, y, 10]); cs.append([b*0.85, b*0.45, b*0.35])
    for y in np.linspace(-0.8, 2.5, 40):
        for z in np.linspace(1, 10, 80):
            s = 0.3 + 0.2*np.sin(z*5)
            gs.append([-3, y, z]); cs.append([0.25, 0.3+s*0.3, 0.5+s*0.2])
    for y in np.linspace(-0.8, 2.5, 40):
        for z in np.linspace(1, 10, 80):
            s = 0.3 + 0.2*np.sin(z*4+1)
            gs.append([3, y, z]); cs.append([0.3+s*0.2, 0.5+s*0.3, 0.25])
    for x in np.linspace(-0.8, 0.8, 25):
        for z in np.linspace(4, 5.5, 25):
            gs.append([x, 0.2, z]); cs.append([0.45, 0.3, 0.2])
    for tx, tz in [(-0.7,4.1),(0.7,4.1),(-0.7,5.4),(0.7,5.4)]:
        for y in np.linspace(-0.8, 0.2, 15):
            gs.append([tx, y, tz]); cs.append([0.35, 0.22, 0.15])
    for _ in range(300):
        p = np.array([-0.3,0.5,4.5]) + np.random.randn(3)*[0.08,0.15,0.08]
        gs.append(p.tolist()); cs.append([0.85, 0.12, 0.1])
    for _ in range(300):
        p = np.array([0.3,0.4,4.8]) + np.random.randn(3)*[0.12,0.1,0.12]
        gs.append(p.tolist()); cs.append([0.12, 0.7, 0.15])
    for _ in range(250):
        p = np.array([0.0,0.55,5.2]) + np.random.randn(3)*[0.06,0.2,0.06]
        gs.append(p.tolist()); cs.append([0.15, 0.2, 0.82])
    for x in np.linspace(-1, 1, 30):
        for y in np.linspace(0.5, 1.8, 20):
            r = 0.5+0.5*np.sin(x*5)*np.cos(y*5)
            g = 0.5+0.5*np.cos(x*3+y*4)
            b = 0.5+0.3*np.sin(x*7-y*2)
            gs.append([x, y, 9.9]); cs.append([r, g, b])

    means = torch.tensor(gs, dtype=torch.float32, device=device)
    colors = torch.tensor(np.clip(cs, 0, 1), dtype=torch.float32, device=device)
    n = len(means)
    quats = torch.zeros(n, 4, device=device); quats[:, 0] = 1
    scales = torch.full((n, 3), 0.04, device=device)
    opacities = torch.full((n,), 0.97, device=device)
    gmap = GaussianMap(n_gaussians=0, device=device)
    gmap.means = nn.Parameter(means)
    gmap.quats = nn.Parameter(quats)
    gmap.scales = nn.Parameter(scales)
    gmap.opacities = nn.Parameter(opacities)
    gmap.colors = nn.Parameter(colors)
    return gmap


def traj(n, r=0.8, cz=5.5):
    poses = []
    for i in range(n):
        a = 2*np.pi*i/n
        x, z = r*np.cos(a), cz + r*0.6*np.sin(a)
        fwd = np.array([0,0,cz]) - np.array([x,0.3,z])
        fwd /= np.linalg.norm(fwd)
        right = np.cross([0,1,0], fwd); right /= np.linalg.norm(right)
        up = np.cross(fwd, right)
        T = np.eye(4)
        T[:3,0], T[:3,1], T[:3,2], T[:3,3] = right, up, fwd, [x, 0.3, z]
        poses.append(T)
    return poses


def render(gmap, pose, K, W, H, device):
    vm = torch.inverse(torch.tensor(pose, dtype=torch.float32, device=device))
    with torch.no_grad():
        r, _, _ = render_gaussians(gmap.means, gmap.quats, gmap.scales,
                                    gmap.opacities, gmap.colors, vm, K, W, H)
    return np.clip(r.cpu().numpy(), 0, 1)


def track(gmap, target, init, K, W, H, device):
    key = gtsam.symbol('t', 0)
    p0 = matrix_to_pose3(init)
    g = gtsam.NonlinearFactorGraph()
    v = gtsam.Values()
    g.addPriorPose3(key, p0, gtsam.noiseModel.Isotropic.Sigma(6, 2.0))
    pix = sample_pixel_indices(H, W, 1024)
    f = GaussianSplatFactor(gmap, target, K, pix, W, H, device)
    g.add(f.as_gtsam_factor(key, gtsam.noiseModel.Isotropic.Sigma(f.n_residuals, 0.3)))
    v.insert(key, p0)
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(8)
    params.setVerbosityLM("SILENT")
    return pose3_to_matrix(gtsam.LevenbergMarquardtOptimizer(g, v, params).optimize().atPose3(key))


def u8(img): return (np.clip(img, 0, 1)*255).astype(np.uint8)
def bgr(img): return cv2.cvtColor(u8(img), cv2.COLOR_RGB2BGR)


def put_text(img, text, pos, color, scale=0.5, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def make_tracking_gif(gt_frames, noisy_frames, tracked_frames, err_init, err_track, outpath):
    """Side-by-side GIF: left=GT, right=before/after tracking with error overlay."""
    import imageio
    H, W = gt_frames[0].shape[:2]
    pad = 4
    frames_gif = []
    for i in range(len(gt_frames)):
        gt = bgr(gt_frames[i])
        noisy = bgr(noisy_frames[i])
        tracked = bgr(tracked_frames[i])

        # Top label bar
        bar_h = 30
        canvas = np.zeros((H + bar_h, W * 3 + pad * 2, 3), dtype=np.uint8)

        # Place frames
        canvas[bar_h:bar_h+H, 0:W] = gt
        canvas[bar_h:bar_h+H, W+pad:2*W+pad] = noisy
        canvas[bar_h:bar_h+H, 2*W+2*pad:3*W+2*pad] = tracked

        # Labels
        put_text(canvas, "Ground Truth", (10, 20), (0, 255, 0), 0.45)
        put_text(canvas, f"Noisy Init ({err_init[i]:.3f}m)", (W+pad+10, 20), (0, 100, 255), 0.45)
        put_text(canvas, f"Tracked ({err_track[i]:.3f}m)", (2*W+2*pad+10, 20), (255, 200, 0), 0.45)

        # Frame number
        put_text(canvas, f"Frame {i}", (W*3+2*pad-90, H+bar_h-8), (200, 200, 200), 0.4)

        frames_gif.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    imageio.mimsave(outpath, frames_gif, duration=0.4, loop=0)
    print(f"   {outpath} ({len(frames_gif)} frames)")


def main():
    device = "cuda"
    H, W = 300, 400
    K_np = np.array([[200,0,200],[0,200,150],[0,0,1]], dtype=np.float64)
    K_t = torch.tensor(K_np, dtype=torch.float32, device=device)
    n_frames = 24
    os.makedirs("assets", exist_ok=True)

    print("=== Building Demo ===\n")

    print("1. Scene...")
    gmap = build_room(device)
    print(f"   {gmap.n_gaussians} Gaussians")

    print("2. Trajectory...")
    gt_poses = traj(n_frames)

    print("3. GT renders...")
    gt_imgs = [render(gmap, p, K_t, W, H, device) for p in gt_poses]

    print("4. Tracking...")
    np.random.seed(42)
    tracked_poses, noisy_poses = [], []
    err_init, err_track = [], []
    for i in range(n_frames):
        noisy = gt_poses[i].copy()
        noisy[:3, 3] += np.random.randn(3) * 0.05
        noisy_poses.append(noisy)
        ei = np.linalg.norm(gt_poses[i][:3,3] - noisy[:3,3])
        est = track(gmap, gt_imgs[i], noisy, K_t, W, H, device)
        et = np.linalg.norm(gt_poses[i][:3,3] - est[:3,3])
        tracked_poses.append(est)
        err_init.append(ei); err_track.append(et)
        print(f"   {i:2d}: {ei:.3f}m -> {et:.3f}m")

    print(f"\n   ATE: {np.mean(err_init):.3f}m -> {np.mean(err_track):.3f}m")

    print("\n5. iSAM2 + loop closure...")
    isam = gtsam.ISAM2(gtsam.ISAM2Params())
    for i in range(n_frames):
        g = gtsam.NonlinearFactorGraph(); v = gtsam.Values()
        k = gtsam.symbol('x', i)
        if i == 0:
            g.addPriorPose3(k, matrix_to_pose3(tracked_poses[0]),
                           gtsam.noiseModel.Isotropic.Sigma(6, 0.01))
        else:
            pk = gtsam.symbol('x', i-1)
            odom = matrix_to_pose3(tracked_poses[i-1]).between(matrix_to_pose3(tracked_poses[i]))
            g.add(gtsam.BetweenFactorPose3(pk, k, odom,
                  gtsam.noiseModel.Isotropic.Sigma(6, 0.05)))
        v.insert(k, matrix_to_pose3(tracked_poses[i]))
        isam.update(g, v)

    lc = gtsam.NonlinearFactorGraph()
    lc.add(gtsam.BetweenFactorPose3(
        gtsam.symbol('x', 0), gtsam.symbol('x', n_frames-1),
        matrix_to_pose3(gt_poses[0]).between(matrix_to_pose3(gt_poses[-1])),
        gtsam.noiseModel.Isotropic.Sigma(6, 0.02)))
    isam.update(lc, gtsam.Values())
    for _ in range(3): isam.update()
    est = isam.calculateEstimate()
    final = {i: pose3_to_matrix(est.atPose3(gtsam.symbol('x', i))) for i in range(n_frames)}
    err_final = [np.linalg.norm(gt_poses[i][:3,3] - final[i][:3,3]) for i in range(n_frames)]
    print(f"   ATE: {np.mean(err_final):.3f}m")

    print("\n6. Rendering all views...")
    noisy_imgs = [render(gmap, p, K_t, W, H, device) for p in noisy_poses]
    tracked_imgs = [render(gmap, p, K_t, W, H, device) for p in tracked_poses]
    final_imgs = [render(gmap, final[i], K_t, W, H, device) for i in range(n_frames)]

    print("\n7. Generating assets...")

    # --- Tracking GIF: GT | Noisy | Tracked ---
    make_tracking_gif(gt_imgs, noisy_imgs, tracked_imgs, err_init, err_track,
                      "assets/demo.gif")

    # --- Trajectory plot ---
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    gt_xz = np.array([p[:3,3] for p in gt_poses])
    noisy_xz = np.array([p[:3,3] for p in noisy_poses])
    final_xz = np.array([final[i][:3,3] for i in range(n_frames)])

    ax1.plot(gt_xz[:,0], gt_xz[:,2], "g-o", label="Ground Truth", ms=6, lw=2.5, zorder=3)
    ax1.plot(noisy_xz[:,0], noisy_xz[:,2], "rx--", label="Noisy Initial", ms=5, lw=1, alpha=0.5)
    ax1.plot(final_xz[:,0], final_xz[:,2], "b-s", label="SplatSLAM", ms=5, lw=2, zorder=2)
    ax1.set_xlabel("X (m)", fontsize=12); ax1.set_ylabel("Z (m)", fontsize=12)
    ax1.set_title(f"Trajectory  —  ATE: {np.mean(err_final):.3f}m", fontsize=13)
    ax1.legend(fontsize=10); ax1.set_aspect("equal"); ax1.grid(alpha=0.3)

    x = np.arange(n_frames)
    w = 0.3
    ax2.bar(x - w, err_init, w, color="salmon", label="Noisy init", alpha=0.8)
    ax2.bar(x, err_track, w, color="royalblue", label="After tracking", alpha=0.9)
    ax2.bar(x + w, err_final, w, color="seagreen", label="After iSAM2", alpha=0.9)
    ax2.set_xlabel("Frame", fontsize=12); ax2.set_ylabel("Error (m)", fontsize=12)
    ax2.set_title("Per-frame Translation Error", fontsize=13)
    ax2.legend(fontsize=10); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("assets/trajectory.png", dpi=150, bbox_inches="tight")
    print(f"   assets/trajectory.png")

    improved = sum(1 for a, b in zip(err_track, err_init) if a < b)
    print(f"\n=== Done: {np.mean(err_init):.3f}m -> {np.mean(err_track):.3f}m -> {np.mean(err_final):.3f}m | {improved}/{n_frames} improved ===")


if __name__ == "__main__":
    main()
