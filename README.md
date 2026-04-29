# gtsam-splatfactors

**Gaussian Splatting meets Factor Graph SLAM** — iSAM2 incremental pose optimization with differentiable rendering factors.

## What is this?

Current 3DGS-SLAM systems (SplaTAM, MonoGS, Photo-SLAM) optimize camera poses via gradient descent through a differentiable rasterizer. They work, but they lack the infrastructure that factor graph SLAM provides: loop closure, incremental Bayes tree updates, proper marginalization, and principled uncertainty estimation.

This library bridges the two worlds. Camera poses live in a GTSAM factor graph (iSAM2), and photometric errors from Gaussian splatting rendering are expressed as GTSAM factors. You get:

- **Loop closure** — add a rendering factor between any two keyframes and iSAM2 corrects all poses
- **Incremental updates** — iSAM2's Bayes tree avoids re-optimizing from scratch on every frame
- **Principled uncertainty** — GTSAM gives covariances on all poses
- **Modularity** — easily combine with IMU preintegration, wheel odometry, GPS, etc.

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
- [x] Loop closure support
- [ ] Analytical Jacobians through gsplat autograd
- [ ] Keyframe selection heuristics
- [ ] Dense depth initialization from monocular depth
- [ ] Benchmarks on TUM-RGBD / Replica

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
