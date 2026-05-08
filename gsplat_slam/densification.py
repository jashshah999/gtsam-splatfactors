"""Gaussian densification and pruning for incremental mapping.

As new keyframes are added, the Gaussian map needs to grow (add new Gaussians
in newly observed regions) and be pruned (remove low-opacity or redundant
Gaussians). This implements an incremental version of 3DGS densification
that works within the SLAM loop.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class IncrementalDensifier:
    """Handles growing and pruning the Gaussian map during SLAM."""

    def __init__(
        self,
        grad_threshold: float = 0.0002,
        opacity_threshold: float = 0.005,
        scale_threshold: float = 0.5,
        max_gaussians: int = 500000,
        densify_every: int = 100,
        prune_every: int = 200,
        device: str = "cuda",
    ):
        self.grad_threshold = grad_threshold
        self.opacity_threshold = opacity_threshold
        self.scale_threshold = scale_threshold
        self.max_gaussians = max_gaussians
        self.densify_every = densify_every
        self.prune_every = prune_every
        self.device = device
        self.step = 0
        self.grad_accum = None
        self.grad_count = None

    def accumulate_gradients(self, means: torch.Tensor):
        """Accumulate position gradients for densification decisions."""
        if means.grad is None:
            return

        grad_norm = means.grad.norm(dim=-1)

        if self.grad_accum is None or len(self.grad_accum) != len(means):
            self.grad_accum = torch.zeros(len(means), device=self.device)
            self.grad_count = torch.zeros(len(means), device=self.device)

        self.grad_accum += grad_norm
        self.grad_count += 1
        self.step += 1

    def should_densify(self) -> bool:
        return self.step > 0 and self.step % self.densify_every == 0

    def should_prune(self) -> bool:
        return self.step > 0 and self.step % self.prune_every == 0

    def densify(self, params: dict) -> dict:
        """Split/clone Gaussians with high gradient accumulation.

        Gaussians with large positional gradients are either:
        - Split (if large scale): replaced by two smaller Gaussians
        - Cloned (if small scale): duplicated with slight offset

        Returns updated params dict.
        """
        if self.grad_accum is None:
            return params

        n = len(params["means"])
        avg_grad = self.grad_accum / (self.grad_count + 1e-8)

        # Find Gaussians that need densification
        high_grad = avg_grad > self.grad_threshold

        if not high_grad.any():
            self._reset_accum()
            return params

        scales = params["scales"].data
        mean_scale = scales.exp().mean(dim=-1)

        # Split: high gradient + large scale
        split_mask = high_grad & (mean_scale > self.scale_threshold)
        # Clone: high gradient + small scale
        clone_mask = high_grad & ~split_mask

        new_means = []
        new_quats = []
        new_scales = []
        new_opacities = []
        new_colors = []

        # Clone: duplicate with slight random offset
        if clone_mask.any():
            clone_idx = clone_mask.nonzero(as_tuple=True)[0]
            n_clone = min(len(clone_idx), self.max_gaussians - n)
            if n_clone > 0:
                clone_idx = clone_idx[:n_clone]
                offset = torch.randn(n_clone, 3, device=self.device) * 0.01
                new_means.append(params["means"].data[clone_idx] + offset)
                new_quats.append(params["quats"].data[clone_idx])
                new_scales.append(params["scales"].data[clone_idx])
                new_opacities.append(params["opacities"].data[clone_idx])
                new_colors.append(params["colors"].data[clone_idx])

        # Split: replace with two smaller Gaussians
        if split_mask.any():
            split_idx = split_mask.nonzero(as_tuple=True)[0]
            n_split = min(len(split_idx), (self.max_gaussians - n) // 2)
            if n_split > 0:
                split_idx = split_idx[:n_split]
                # Each split Gaussian becomes two at offset positions with halved scale
                for _ in range(2):
                    offset = torch.randn(n_split, 3, device=self.device) * mean_scale[split_idx, None] * 0.5
                    new_means.append(params["means"].data[split_idx] + offset)
                    new_quats.append(params["quats"].data[split_idx])
                    new_scales.append(params["scales"].data[split_idx] - 0.7)  # halve scale (log space)
                    new_opacities.append(params["opacities"].data[split_idx])
                    new_colors.append(params["colors"].data[split_idx])

        if new_means:
            for key, new_list in [
                ("means", new_means), ("quats", new_quats),
                ("scales", new_scales), ("opacities", new_opacities),
                ("colors", new_colors),
            ]:
                combined = torch.cat([params[key].data] + new_list)
                params[key] = nn.Parameter(combined)

        self._reset_accum()
        return params

    def prune(self, params: dict) -> dict:
        """Remove low-opacity and oversized Gaussians.

        Returns updated params dict with pruned Gaussians removed.
        """
        opacities = params["opacities"].data.sigmoid()
        scales = params["scales"].data.exp()

        # Keep Gaussians that are: visible (high opacity) and reasonable size
        keep = (opacities > self.opacity_threshold) & (scales.max(dim=-1).values < 2.0)

        # Also remove NaN/Inf
        keep &= torch.isfinite(params["means"].data).all(dim=-1)

        if keep.all():
            return params

        n_before = len(params["means"])
        for key in params:
            params[key] = nn.Parameter(params[key].data[keep])
        n_after = len(params["means"])

        if n_before != n_after:
            print(f"    Pruned {n_before - n_after} Gaussians ({n_before} → {n_after})")

        self._reset_accum()
        return params

    def _reset_accum(self):
        self.grad_accum = None
        self.grad_count = None


def add_gaussians_from_new_view(
    params: dict,
    image: np.ndarray,
    depth: np.ndarray,
    pose_c2w: np.ndarray,
    K: np.ndarray,
    existing_means: torch.Tensor,
    stride: int = 8,
    min_dist: float = 0.05,
    device: str = "cuda",
) -> dict:
    """Add new Gaussians from a new viewpoint where the map has gaps.

    Only adds Gaussians in regions not already covered by existing ones.
    """
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    rows = np.arange(0, H, stride)
    cols = np.arange(0, W, stride)
    rr, cc = np.meshgrid(rows, cols, indexing='ij')
    rr, cc = rr.flatten(), cc.flatten()
    d = depth[rr, cc]
    valid = (d > 0.05) & (d < 10.0) & np.isfinite(d)
    rr, cc, d = rr[valid], cc[valid], d[valid]

    if len(d) == 0:
        return params

    # Back-project to world frame
    x_cam = (cc - cx) * d / fx
    y_cam = (rr - cy) * d / fy
    pts_cam = np.stack([x_cam, y_cam, d, np.ones_like(d)], axis=1)
    pts_world = (pose_c2w @ pts_cam.T).T[:, :3]

    # Filter: only keep points far from existing Gaussians
    new_pts = torch.tensor(pts_world, dtype=torch.float32, device=device)
    if len(existing_means) > 0:
        # Subsample existing for speed
        n_existing = min(len(existing_means), 10000)
        idx = torch.randperm(len(existing_means))[:n_existing]
        existing_sub = existing_means[idx]

        # Check minimum distance
        dists = torch.cdist(new_pts, existing_sub).min(dim=1).values
        far_enough = dists > min_dist
        new_pts = new_pts[far_enough]
        rr_valid = rr[far_enough.cpu().numpy()] if len(rr) > len(new_pts) else rr[:len(new_pts)]
        cc_valid = cc[far_enough.cpu().numpy()] if len(cc) > len(new_pts) else cc[:len(new_pts)]
    else:
        rr_valid = rr
        cc_valid = cc

    if len(new_pts) == 0:
        return params

    # Get colors
    colors_np = image[rr_valid[:len(new_pts)], cc_valid[:len(new_pts)]]
    if colors_np.max() > 1:
        colors_np = colors_np.astype(np.float32) / 255.0
    new_colors = torch.tensor(colors_np, dtype=torch.float32, device=device)

    n_new = len(new_pts)
    new_quats = torch.nn.functional.normalize(torch.randn(n_new, 4, device=device), dim=-1)
    avg_depth = float(np.median(d))
    pixel_size = avg_depth * stride / ((fx + fy) / 2)
    init_scale = float(np.clip(pixel_size * 2, 0.01, 0.1))
    new_scales = torch.full((n_new, 3), np.log(init_scale), device=device)
    new_opacities = torch.full((n_new,), 2.0, device=device)

    for key, new_data in [
        ("means", new_pts), ("quats", new_quats), ("scales", new_scales),
        ("opacities", new_opacities), ("colors", new_colors),
    ]:
        params[key] = nn.Parameter(torch.cat([params[key].data, new_data]))

    print(f"    Added {n_new} new Gaussians from new view")
    return params
