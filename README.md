# gtsam-splatfactors

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**3DGS-SLAM with a real SLAM backend.** Loop closure, IMU fusion, uncertainty -- things SplaTAM and MonoGS can't do.

## The Problem

Every 3DGS-SLAM system (SplaTAM, MonoGS, Photo-SLAM, GS-ICP-SLAM) optimizes poses via **gradient descent** through a differentiable rasterizer. This works for tracking, but fundamentally cannot do:

| Capability | Gradient Descent | Factor Graph (this repo) |
|-----------|-----------------|--------------------------|
| Loop closure (correct all poses when revisiting) | No | Yes |
| Incremental updates (O(log n) per frame) | No (re-optimize all) | Yes (iSAM2 Bayes tree) |
| Pose uncertainty / covariance | No | Yes |
| IMU / GPS / wheel odometry fusion | No | Yes (add factors) |
| Robust outlier rejection | No | Yes (Cauchy/Huber) |

These aren't nice-to-haves -- they're required for any SLAM system that operates beyond a single room. SplaTAM drifts. MonoGS drifts. This repo fixes that.

## How It Works

Camera poses live in a GTSAM factor graph. Photometric errors from gsplat rendering are expressed as GTSAM factors. You get rendering-quality tracking AND all the factor graph infrastructure:

```
Camera poses (Pose3)          Gaussian map (means, colors, ...)
        |                              |
   +----------+                   +----------+
   |  iSAM2   |<-- SplatFactor -->|  gsplat  |
   |  (GTSAM) |   (photometric    | renderer |
   +----------+    residual)      +----------+
        |
   Odometry factors
   Loop closure factors (DINOv2 + photometric verification)
   IMU preintegration factors
   Prior factors
```

When a loop closure is detected, iSAM2 corrects ALL downstream poses in O(log n) -- not gradient descent on the whole trajectory.

## Demo

Synthetic room scene (22k Gaussians) — camera tracks a circular trajectory with 5cm noise injected into initial poses. Photometric factors correct the poses via LM optimization, then iSAM2 enforces global consistency with loop closure.

**Tracking: Ground Truth | Noisy Initial | After Optimization**

![Demo](assets/demo.gif)

**Trajectory + per-frame error reduction**

![Trajectory](assets/trajectory.png)

ATE: **0.060m** (noisy init) → **0.006m** (after photometric tracking) → **0.007m** (after iSAM2 + loop closure). 23/24 frames improved.

> Synthetic scene shows the factor graph pipeline working end-to-end. See TUM-RGBD results below for real-data evaluation.

**TUM-RGBD fr1/desk** — trained 148k Gaussians on 10 keyframes, tracked remaining 40 frames:

![TUM Results](assets/tum_fr1_desk.png)

First tracked frame: **6.1cm** error. Drift increases as camera moves beyond initial map coverage. Trained Gaussian render vs GT:

<img src="assets/tum_trained_render.png" width="320"> <img src="assets/tum_gt_frame0.png" width="320">

The tracking works where the map has coverage. Reducing drift requires denser incremental mapping — see roadmap.

Run the demo yourself: `python examples/make_demo_video.py` (requires CUDA GPU).

## Architecture

```
Camera poses (Pose3)          Gaussian map (means, colors, ...)
        │                              │
   ┌────▼────┐                    ┌────▼────┐
   │  iSAM2  │◄── SplatFactor ──►│  gsplat  │
   │  (GTSAM)│    (photometric    │ renderer │
   └─────────┘     residual)      └──────────┘
        │
   Odometry factors
   Loop closure factors
   Prior factors
```

**Key design:** Poses are optimized via GTSAM (factor graph). Gaussians are optimized separately via Adam (too many parameters for factor graphs). Alternating optimization keeps both consistent.

## Installation

```bash
pip install gtsam gsplat torch opencv-python
pip install -e .
```

Requires CUDA for the gsplat rasterizer.

## Quick start

```python
from gsplat_slam import SplatSLAM
import numpy as np

# Camera intrinsics
K = np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]], dtype=np.float64)

slam = SplatSLAM(K=K, W=640, H=480, device="cuda")

# Add keyframes (RGB image + optional depth)
for image, depth in dataset:
    pose = slam.add_keyframe(image, depth)
    print(f"Estimated pose:\n{pose}")

# Add loop closure when detected
slam.add_loop_closure(idx_from=0, idx_to=50, relative_pose=T_0_50)

# Get all corrected poses
poses = slam.get_all_poses()
```

## The `GaussianSplatFactor`

The core contribution is `GaussianSplatFactor` — a GTSAM-compatible factor that:

1. Renders the Gaussian map from a candidate camera pose using gsplat
2. Computes photometric residuals at sampled pixel locations
3. Provides Jacobians for GTSAM's optimizer (numerical, through the Lie algebra)

```python
from gsplat_slam import GaussianSplatFactor

factor = GaussianSplatFactor(
    gaussian_map=my_map,
    target_image=keyframe_rgb,
    K=intrinsics,
    pixel_indices=sampled_pixels,
    W=640, H=480,
)

# Use directly
residual, jacobian = factor.evaluate(pose)

# Or add to GTSAM graph
gtsam_factor = factor.as_gtsam_factor(pose_key, noise_model)
graph.add(gtsam_factor)
```

## Status

This is early-stage research code. Phase 1 (core factor + SLAM pipeline) is implemented. Contributions welcome.

- [x] `GaussianSplatFactor` with numerical Jacobians
- [x] `GaussianMap` with point cloud initialization
- [x] `SplatSLAM` incremental pipeline with iSAM2
- [x] Loop closure support (manual + automatic DINOv2 detection)
- [x] TUM-RGBD dataset loader and evaluation script
- [x] Gaussian map training with L1 loss (converges to 0.04 on synthetic, 0.11 on TUM)
- [x] Monocular depth initialization (Depth Anything V2, ZoeDepth)
- [x] Keyframe selection (translation + rotation + overlap heuristics)
- [x] Automatic loop detection via DINOv2 + photometric verification
- [x] COLMAP / nerfstudio / PLY export
- [ ] Densification (gsplat DefaultStrategy — currently causes FPE, needs debugging)
- [ ] Analytical Jacobians through gsplat autograd (replace numerical, ~10x faster)

## How it compares

| Feature | SplaTAM | MonoGS | **gtsam-splatfactors** |
|---|---|---|---|
| Pose optimization | Gradient descent | Gradient descent | **iSAM2 (Bayes tree)** |
| Loop closure | No | No | **Yes** |
| Incremental updates | No (re-optimize) | No | **Yes** |
| Uncertainty estimates | No | No | **Yes (covariances)** |
| IMU/odometry fusion | No | No | **Yes (add factors)** |
| Rendering quality | Good | Good | Good (same gsplat) |

## Citation

If you use this in your research:

```bibtex
@software{gtsam_splatfactors,
  author = {Shah, Jash},
  title = {gtsam-splatfactors: Gaussian Splatting meets Factor Graph SLAM},
  year = {2026},
  url = {https://github.com/jashshah999/gtsam-splatfactors}
}
```

## License

MIT
