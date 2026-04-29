"""Gaussian map trainer using gsplat's DefaultStrategy for densification."""

import torch
import torch.nn as nn
import numpy as np
from gsplat import rasterization, DefaultStrategy


class GaussianMapper:
    """Trains and maintains a Gaussian splat map from RGB-D keyframes.

    Uses gsplat's DefaultStrategy for automatic densification (split/clone/prune).
    """

    def __init__(
        self,
        device: str = "cuda",
        lr_means: float = 0.0005,
        lr_colors: float = 0.02,
        lr_scales: float = 0.01,
        lr_opacities: float = 0.05,
        lr_quats: float = 0.001,
        refine_start: int = 200,
        refine_every: int = 100,
        reset_every: int = 1000,
    ):
        self.device = device
        self.params = None
        self.optimizers = None
        self.strategy = DefaultStrategy(
            refine_start_iter=refine_start,
            refine_every=refine_every,
            reset_every=reset_every,
            grow_grad2d=0.0002,
            prune_opa=0.005,
            prune_scale3d=0.1,
            verbose=False,
        )
        self.state = None
        self.global_step = 0
        self.lr_config = {
            "means": lr_means, "colors": lr_colors, "scales": lr_scales,
            "opacities": lr_opacities, "quats": lr_quats,
        }

    def init_from_rgbd(self, rgb: np.ndarray, depth: np.ndarray,
                       pose: np.ndarray, K: np.ndarray, stride: int = 4):
        """Initialize Gaussians from an RGB-D frame."""
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
        pts_cam = np.stack([x_cam, y_cam, d, np.ones_like(d)], axis=1)
        pts_world = (pose @ pts_cam.T).T[:, :3]
        colors_np = rgb[rr, cc]

        means = torch.tensor(pts_world, dtype=torch.float32, device=self.device)
        colors = torch.tensor(colors_np, dtype=torch.float32, device=self.device)
        n = len(means)

        # Compute initial scale from depth and stride: each Gaussian should cover
        # roughly one pixel-stride in world space
        avg_depth = np.median(d)
        pixel_size = avg_depth * stride / ((fx + fy) / 2)
        init_scale = max(pixel_size, 0.005)

        if self.params is None:
            self.params = {
                "means": nn.Parameter(means),
                "quats": nn.Parameter(torch.nn.functional.normalize(
                    torch.randn(n, 4, device=self.device), dim=-1)),
                "scales": nn.Parameter(torch.full((n, 3), init_scale, device=self.device)),
                "opacities": nn.Parameter(torch.full((n,), 2.0, device=self.device)),
                "colors": nn.Parameter(colors),
            }
        else:
            # Append new Gaussians
            n_new = len(means)
            for k in self.params:
                old = self.params[k].data
                if k == "means":
                    new = means
                elif k == "colors":
                    new = colors
                elif k == "quats":
                    new = torch.nn.functional.normalize(
                        torch.randn(n_new, 4, device=self.device), dim=-1)
                elif k == "scales":
                    new = torch.full((n_new, 3), init_scale, device=self.device)
                elif k == "opacities":
                    new = torch.full((n_new,), 2.0, device=self.device)
                self.params[k] = nn.Parameter(torch.cat([old, new]))

        self._rebuild_optimizers()
        self.state = self.strategy.initialize_state(scene_scale=1.0)

    def _rebuild_optimizers(self):
        self.optimizers = {}
        for k, p in self.params.items():
            lr = self.lr_config.get(k, 0.001)
            self.optimizers[k] = torch.optim.Adam([p], lr=lr)

    @property
    def n_gaussians(self):
        if self.params is None:
            return 0
        return len(self.params["means"])

    def train_step(self, rgb_target: torch.Tensor, viewmat: torch.Tensor,
                   K: torch.Tensor, W: int, H: int) -> float:
        """One training step: render, compute loss, backprop, densify."""
        for opt in self.optimizers.values():
            opt.zero_grad()

        rendered, alpha, info = rasterization(
            means=self.params["means"],
            quats=self.params["quats"],
            scales=self.params["scales"],
            opacities=self.params["opacities"].sigmoid(),
            colors=self.params["colors"],
            viewmats=viewmat[None],
            Ks=K[None],
            width=W, height=H,
            packed=False,
        )

        l1 = torch.nn.functional.l1_loss(rendered[0], rgb_target)
        ssim_val = 1.0 - self._ssim(rendered[0], rgb_target)
        loss = 0.8 * l1 + 0.2 * ssim_val

        self.strategy.step_pre_backward(
            self.params, self.optimizers, self.state, self.global_step, info)

        loss.backward()

        self.strategy.step_post_backward(
            self.params, self.optimizers, self.state, self.global_step, info, packed=False)

        for opt in self.optimizers.values():
            opt.step()

        self.global_step += 1
        return loss.item()

    def train_on_frames(self, frames: list, K_torch: torch.Tensor,
                        W: int, H: int, n_iters: int = 300,
                        log_every: int = 50):
        """Train the Gaussian map on a list of keyframes."""
        for step in range(n_iters):
            total_loss = 0
            for frame in frames:
                rgb_t = torch.tensor(frame["rgb"], dtype=torch.float32, device=self.device)
                vm = torch.inverse(torch.tensor(
                    frame["pose"], dtype=torch.float32, device=self.device))
                vm[1] = -vm[1]  # Y-flip for gsplat
                loss = self.train_step(rgb_t, vm, K_torch, W, H)
                total_loss += loss
            avg = total_loss / len(frames)
            if step % log_every == 0:
                print(f"    Step {step:4d}: loss={avg:.4f}  n_gs={self.n_gaussians}")
        return avg

    def render(self, pose: np.ndarray, K_torch: torch.Tensor, W: int, H: int):
        """Render from a pose. Returns (H, W, 3) numpy array."""
        vm = torch.inverse(torch.tensor(pose, dtype=torch.float32, device=self.device))
        vm[1] = -vm[1]
        with torch.no_grad():
            rendered, _, _ = rasterization(
                means=self.params["means"],
                quats=self.params["quats"],
                scales=self.params["scales"],
                opacities=self.params["opacities"].sigmoid(),
                colors=self.params["colors"],
                viewmats=vm[None], Ks=K_torch[None],
                width=W, height=H, packed=False)
        return np.clip(rendered[0].cpu().numpy(), 0, 1)

    @staticmethod
    def _ssim(img1, img2, window_size=11):
        """Simple SSIM approximation."""
        C1, C2 = 0.01**2, 0.03**2
        mu1 = torch.nn.functional.avg_pool2d(
            img1.permute(2, 0, 1).unsqueeze(0), window_size, 1, window_size//2)
        mu2 = torch.nn.functional.avg_pool2d(
            img2.permute(2, 0, 1).unsqueeze(0), window_size, 1, window_size//2)
        mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1*mu2
        sigma1_sq = torch.nn.functional.avg_pool2d(
            (img1.permute(2, 0, 1).unsqueeze(0))**2, window_size, 1, window_size//2) - mu1_sq
        sigma2_sq = torch.nn.functional.avg_pool2d(
            (img2.permute(2, 0, 1).unsqueeze(0))**2, window_size, 1, window_size//2) - mu2_sq
        sigma12 = torch.nn.functional.avg_pool2d(
            img1.permute(2, 0, 1).unsqueeze(0) * img2.permute(2, 0, 1).unsqueeze(0),
            window_size, 1, window_size//2) - mu12
        ssim_map = ((2*mu12 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()
