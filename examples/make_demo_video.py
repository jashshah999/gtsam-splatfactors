"""Generate demo GIF and trajectory plot for README.

Produces:
  assets/demo.gif        — GT | Noisy | Tracked side-by-side
  assets/trajectory.png  — trajectory + error bars

Usage:  python examples/make_demo_video.py
"""

import numpy as np
import torch
import torch.nn as nn
import cv2
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gsplat_slam.map import GaussianMap
from gsplat_slam.renderer import render_gaussians, sample_pixel_indices
from gsplat_slam.factor import GaussianSplatFactor
from gsplat_slam.pose_utils import matrix_to_pose3, pose3_to_matrix
import gtsam


def build_room(device="cuda"):
    np.random.seed(0)
    gs, cs = [], []

    # Dense wood floor with grain
    for x in np.linspace(-3.5, 3.5, 120):
        for z in np.linspace(1, 12, 180):
            grain = 0.4 + 0.25*np.sin(x*10 + 0.3*z) + 0.1*np.sin(z*5) + np.random.randn()*0.02
            gs.append([x, -0.8, z]); cs.append([grain*0.95, grain*0.65, grain*0.35])

    # Ceiling
    for x in np.linspace(-3.5, 3.5, 80):
        for z in np.linspace(1, 12, 120):
            gs.append([x, 2.8, z]); cs.append([0.92, 0.90, 0.85])

    # Back wall: brick
    for x in np.linspace(-3.5, 3.5, 100):
        for y in np.linspace(-0.8, 2.8, 60):
            bx, by = int(x*3 + 100), int(y*5 + 100)
            mortar = 1.0 if (bx%3==0 or by%4==0) else 0.0
            brick = 0.55 + 0.15*((bx//3 + by//4) % 2)
            c = [0.85, 0.82, 0.75] if mortar > 0.5 else [brick*0.9, brick*0.5, brick*0.38]
            gs.append([x, y, 12]); cs.append(c)

    # Left wall: blue with wainscoting
    for y in np.linspace(-0.8, 2.8, 50):
        for z in np.linspace(1, 12, 100):
            if y < 0.3:
                c = [0.35, 0.25, 0.18]  # dark wood wainscoting
            else:
                v = 0.02 * np.sin(z*8 + y*6)
                c = [0.45+v, 0.55+v, 0.72+v]
            gs.append([-3.5, y, z]); cs.append(c)

    # Right wall: warm
    for y in np.linspace(-0.8, 2.8, 50):
        for z in np.linspace(1, 12, 100):
            if y < 0.3:
                c = [0.35, 0.25, 0.18]
            else:
                v = 0.02 * np.sin(z*6 + y*8)
                c = [0.78+v, 0.65+v, 0.50+v]
            gs.append([3.5, y, z]); cs.append(c)

    # Table
    for x in np.linspace(-1, 1, 35):
        for z in np.linspace(5, 7, 35):
            gs.append([x, 0.1, z]); cs.append([0.50, 0.35, 0.22])
    for tx, tz in [(-0.9,5.1),(0.9,5.1),(-0.9,6.9),(0.9,6.9)]:
        for y in np.linspace(-0.8, 0.1, 12):
            gs.append([tx, y, tz]); cs.append([0.40, 0.28, 0.18])

    # Red teapot on table
    for _ in range(400):
        theta = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0, 0.18)
        h = np.random.uniform(0, 0.3)
        gs.append([-0.4+r*np.cos(theta), 0.25+h, 5.8+r*np.sin(theta)])
        cs.append([0.82+np.random.randn()*0.03, 0.15, 0.12])

    # Green plant
    for _ in range(500):
        base = np.array([0.5, 0.2, 6.2])
        h = np.random.exponential(0.15)
        spread = 0.08 + h*0.3
        p = base + np.array([np.random.randn()*spread, h, np.random.randn()*spread])
        gs.append(p.tolist()); cs.append([0.1+np.random.rand()*0.15, 0.55+np.random.rand()*0.2, 0.1])

    # Blue vase
    for _ in range(300):
        h = np.random.uniform(0, 0.35)
        r = 0.06 + 0.04*np.sin(h*8)
        theta = np.random.uniform(0, 2*np.pi)
        gs.append([0.0+r*np.cos(theta), 0.2+h, 6.5+r*np.sin(theta)])
        cs.append([0.15, 0.25, 0.75+np.random.randn()*0.05])

    # Painting on back wall
    for x in np.linspace(-1.2, 1.2, 40):
        for y in np.linspace(0.8, 2.2, 30):
            r = 0.5 + 0.4*np.sin(x*4)*np.cos(y*3)
            g = 0.4 + 0.3*np.cos(x*5 + y*4)
            b = 0.5 + 0.4*np.sin(x*3 - y*6)
            gs.append([x, y, 11.95]); cs.append([r, g, b])
    # Frame
    for x in np.linspace(-1.3, 1.3, 50):
        gs.append([x, 0.75, 11.93]); cs.append([0.3, 0.2, 0.1])
        gs.append([x, 2.25, 11.93]); cs.append([0.3, 0.2, 0.1])
    for y in np.linspace(0.75, 2.25, 30):
        gs.append([-1.3, y, 11.93]); cs.append([0.3, 0.2, 0.1])
        gs.append([1.3, y, 11.93]); cs.append([0.3, 0.2, 0.1])

    # Bookshelf on left wall
    for shelf_y in [0.3, 0.9, 1.5]:
        for z in np.linspace(8, 10, 20):
            gs.append([-3.45, shelf_y, z]); cs.append([0.45, 0.3, 0.2])
        # Books
        for z in np.linspace(8.1, 9.9, 15):
            h = np.random.uniform(0.15, 0.45)
            col = np.random.rand(3) * 0.5 + 0.3
            for dy in np.linspace(0, h, 5):
                gs.append([-3.42, shelf_y+0.05+dy, z]); cs.append(col.tolist())

    means = torch.tensor(gs, dtype=torch.float32, device=device)
    colors = torch.tensor(np.clip(cs, 0, 1), dtype=torch.float32, device=device)
    n = len(means)
    quats = torch.zeros(n, 4, device=device); quats[:, 0] = 1
    scales = torch.full((n, 3), 0.035, device=device)
    opacities = torch.full((n,), 0.97, device=device)
    gmap = GaussianMap(n_gaussians=0, device=device)
    gmap.means = nn.Parameter(means)
    gmap.quats = nn.Parameter(quats)
    gmap.scales = nn.Parameter(scales)
    gmap.opacities = nn.Parameter(opacities)
    gmap.colors = nn.Parameter(colors)
    return gmap


def make_traj(n, r=1.2, cz=7.0):
    poses = []
    for i in range(n):
        a = 2*np.pi*i/n
        x, z = r*np.cos(a), cz + r*0.7*np.sin(a)
        fwd = np.array([0, 0.5, cz]) - np.array([x, 0.5, z])
        fwd /= np.linalg.norm(fwd)
        right = np.cross([0, 1, 0], fwd); right /= np.linalg.norm(right)
        up = np.cross(fwd, right)
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, up, fwd, [x, 0.5, z]
        poses.append(T)
    return poses


def render_at(gmap, pose, K, W, H, device):
    vm = torch.inverse(torch.tensor(pose, dtype=torch.float32, device=device))
    with torch.no_grad():
        r, _, _ = render_gaussians(gmap.means, gmap.quats, gmap.scales,
                                    gmap.opacities, gmap.colors, vm, K, W, H)
    return np.clip(r.cpu().numpy(), 0, 1)


def track_frame(gmap, target, init, K, W, H, device):
    key = gtsam.symbol('t', 0)
    p0 = matrix_to_pose3(init)
    g = gtsam.NonlinearFactorGraph(); v = gtsam.Values()
    g.addPriorPose3(key, p0, gtsam.noiseModel.Isotropic.Sigma(6, 2.0))
    pix = sample_pixel_indices(H, W, 1536)
    f = GaussianSplatFactor(gmap, target, K, pix, W, H, device)
    g.add(f.as_gtsam_factor(key, gtsam.noiseModel.Isotropic.Sigma(f.n_residuals, 0.3)))
    v.insert(key, p0)
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(10)
    params.setVerbosityLM("SILENT")
    return pose3_to_matrix(gtsam.LevenbergMarquardtOptimizer(g, v, params).optimize().atPose3(key))


def u8(img): return (np.clip(img, 0, 1)*255).astype(np.uint8)
def to_bgr(img): return cv2.cvtColor(u8(img), cv2.COLOR_RGB2BGR)


def put(img, text, pos, color, scale=0.45, thick=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main():
    device = "cuda"
    H, W = 360, 480  # higher res
    K_np = np.array([[280, 0, 240], [0, 280, 180], [0, 0, 1]], dtype=np.float64)
    K_t = torch.tensor(K_np, dtype=torch.float32, device=device)
    n_frames = 24
    os.makedirs("assets", exist_ok=True)

    print("=== Building Demo ===\n")
    print("1. Scene..."); gmap = build_room(device); print(f"   {gmap.n_gaussians} Gaussians")
    print("2. Trajectory..."); gt_poses = make_traj(n_frames)
    print("3. Rendering GT..."); gt_imgs = [render_at(gmap, p, K_t, W, H, device) for p in gt_poses]

    print("4. Tracking...")
    np.random.seed(42)
    tracked, noisy_poses, err_i, err_t = [], [], [], []
    for i in range(n_frames):
        noisy = gt_poses[i].copy(); noisy[:3, 3] += np.random.randn(3)*0.04
        noisy_poses.append(noisy)
        ei = np.linalg.norm(gt_poses[i][:3,3] - noisy[:3,3])
        est = track_frame(gmap, gt_imgs[i], noisy, K_t, W, H, device)
        et = np.linalg.norm(gt_poses[i][:3,3] - est[:3,3])
        tracked.append(est); err_i.append(ei); err_t.append(et)
        print(f"   {i:2d}: {ei:.3f}m -> {et:.3f}m {'OK' if et<ei else '!!'}")

    print(f"\n   ATE: {np.mean(err_i):.3f}m -> {np.mean(err_t):.3f}m")

    print("\n5. iSAM2...")
    isam = gtsam.ISAM2(gtsam.ISAM2Params())
    for i in range(n_frames):
        g = gtsam.NonlinearFactorGraph(); v = gtsam.Values(); k = gtsam.symbol('x', i)
        if i == 0:
            g.addPriorPose3(k, matrix_to_pose3(tracked[0]), gtsam.noiseModel.Isotropic.Sigma(6, 0.01))
        else:
            odom = matrix_to_pose3(tracked[i-1]).between(matrix_to_pose3(tracked[i]))
            g.add(gtsam.BetweenFactorPose3(gtsam.symbol('x',i-1), k, odom, gtsam.noiseModel.Isotropic.Sigma(6, 0.04)))
        v.insert(k, matrix_to_pose3(tracked[i])); isam.update(g, v)
    lc = gtsam.NonlinearFactorGraph()
    lc.add(gtsam.BetweenFactorPose3(gtsam.symbol('x',0), gtsam.symbol('x',n_frames-1),
        matrix_to_pose3(gt_poses[0]).between(matrix_to_pose3(gt_poses[-1])),
        gtsam.noiseModel.Isotropic.Sigma(6, 0.02)))
    isam.update(lc, gtsam.Values())
    for _ in range(3): isam.update()
    est = isam.calculateEstimate()
    final = {i: pose3_to_matrix(est.atPose3(gtsam.symbol('x', i))) for i in range(n_frames)}
    err_f = [np.linalg.norm(gt_poses[i][:3,3]-final[i][:3,3]) for i in range(n_frames)]
    print(f"   ATE after LC: {np.mean(err_f):.3f}m")

    print("\n6. Rendering views...")
    noisy_imgs = [render_at(gmap, p, K_t, W, H, device) for p in noisy_poses]
    tracked_imgs = [render_at(gmap, p, K_t, W, H, device) for p in tracked]

    print("7. Building GIF...")
    import imageio
    frames = []
    pad = 3
    for i in range(n_frames):
        gt_f = to_bgr(gt_imgs[i])
        noisy_f = to_bgr(noisy_imgs[i])
        track_f = to_bgr(tracked_imgs[i])

        bar_h = 28
        canvas = np.zeros((H + bar_h, W*3 + pad*2, 3), dtype=np.uint8)
        canvas[bar_h:, :W] = gt_f
        canvas[bar_h:, W+pad:2*W+pad] = noisy_f
        canvas[bar_h:, 2*W+2*pad:] = track_f

        put(canvas, "Ground Truth", (8, 20), (0, 255, 0))
        put(canvas, f"Noisy Init ({err_i[i]:.3f}m)", (W+pad+8, 20), (80, 120, 255))
        put(canvas, f"Tracked ({err_t[i]:.3f}m)", (2*W+2*pad+8, 20), (0, 220, 255))
        put(canvas, f"Frame {i}/{n_frames-1}", (W*3+2*pad-120, H+bar_h-8), (180, 180, 180), 0.4)

        frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    imageio.mimsave("assets/demo.gif", frames, duration=0.35, loop=0)
    print("   assets/demo.gif")

    print("8. Trajectory plot...")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    gt_xz = np.array([p[:3,3] for p in gt_poses])
    n_xz = np.array([p[:3,3] for p in noisy_poses])
    f_xz = np.array([final[i][:3,3] for i in range(n_frames)])
    ax1.plot(gt_xz[:,0], gt_xz[:,2], "g-o", label="Ground Truth", ms=6, lw=2.5, zorder=3)
    ax1.plot(n_xz[:,0], n_xz[:,2], "rx--", label="Noisy Init", ms=4, lw=0.8, alpha=0.5)
    ax1.plot(f_xz[:,0], f_xz[:,2], "b-s", label="SplatSLAM", ms=5, lw=2, zorder=2)
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Z (m)")
    ax1.set_title(f"Trajectory (ATE: {np.mean(err_f):.3f}m)", fontsize=13)
    ax1.legend(); ax1.set_aspect("equal"); ax1.grid(alpha=0.3)
    x = np.arange(n_frames); w = 0.28
    ax2.bar(x-w, err_i, w, color="salmon", label="Noisy", alpha=0.8)
    ax2.bar(x, err_t, w, color="royalblue", label="Tracked", alpha=0.9)
    ax2.bar(x+w, err_f, w, color="seagreen", label="iSAM2+LC", alpha=0.9)
    ax2.set_xlabel("Frame"); ax2.set_ylabel("Error (m)")
    ax2.set_title("Per-frame Error", fontsize=13); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("assets/trajectory.png", dpi=150, bbox_inches="tight")
    print("   assets/trajectory.png")

    imp = sum(1 for a, b in zip(err_t, err_i) if a < b)
    print(f"\n=== {np.mean(err_i):.3f}m -> {np.mean(err_t):.3f}m -> {np.mean(err_f):.3f}m | {imp}/{n_frames} improved ===")

if __name__ == "__main__":
    main()
