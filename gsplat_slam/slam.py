"""Incremental 3DGS-SLAM pipeline using iSAM2 and GaussianSplatFactor."""

import numpy as np

try:
    import gtsam
    HAS_GTSAM = True
except ImportError:
    HAS_GTSAM = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from gsplat_slam.factor import GaussianSplatFactor
from gsplat_slam.map import GaussianMap
from gsplat_slam.renderer import sample_pixel_indices
from gsplat_slam.pose_utils import matrix_to_pose3, pose3_to_matrix


class SplatSLAM:
    """Incremental 3DGS-SLAM with iSAM2 backend.

    Alternating optimization:
      1. Track: fix Gaussians, optimize new pose via iSAM2
      2. Map:   fix poses, optimize Gaussians via Adam
    """

    def __init__(
        self,
        K: np.ndarray,
        W: int,
        H: int,
        n_pixel_samples: int = 512,
        device: str = "cuda",
        isam2_params: "gtsam.ISAM2Params | None" = None,
        mapping_iters: int = 30,
        mapping_lr: float = 0.01,
        photo_sigma: float = 1.0,
        odom_sigma: float = 0.1,
        tracking_iters: int = 3,
    ):
        assert HAS_GTSAM, "gtsam required"
        assert HAS_TORCH, "torch required"

        self.K_np = K.astype(np.float64)
        self.K_torch = torch.tensor(K, dtype=torch.float32, device=device)
        self.W = W
        self.H = H
        self.n_pixel_samples = n_pixel_samples
        self.device = device
        self.mapping_iters = mapping_iters
        self.mapping_lr = mapping_lr
        self.photo_sigma = photo_sigma
        self.odom_sigma = odom_sigma
        self.tracking_iters = tracking_iters

        if isam2_params is None:
            isam2_params = gtsam.ISAM2Params()
        self.isam2 = gtsam.ISAM2(isam2_params)

        self.gaussian_map = GaussianMap(n_gaussians=0, device=device)
        self.keyframe_count = 0
        self.poses = {}  # key -> gtsam.Pose3
        self.rng = np.random.default_rng(42)

    def _pose_key(self, idx: int):
        return gtsam.symbol('x', idx)

    def add_keyframe(
        self,
        image: np.ndarray,       # (H, W, 3) float [0,1]
        depth: np.ndarray | None = None,  # (H, W) float, meters
        init_pose: np.ndarray | None = None,  # (4, 4) initial guess
        prior_sigma: float | None = None,
    ):
        """Add a new keyframe to the SLAM system.

        Args:
            image: RGB image as float [0, 1].
            depth: Optional depth map for initializing new Gaussians.
            init_pose: 4x4 initial pose guess. Identity if None.
            prior_sigma: If set, adds a prior factor on this pose.

        Returns:
            Optimized pose as a 4x4 numpy matrix.
        """
        idx = self.keyframe_count
        key = self._pose_key(idx)
        self.keyframe_count += 1

        if init_pose is None:
            init_pose = np.eye(4)
        pose_init = matrix_to_pose3(init_pose)

        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()

        # First keyframe: add a strong prior
        if idx == 0 or prior_sigma is not None:
            sigma = prior_sigma if prior_sigma is not None else 0.01
            prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, sigma)
            graph.addPriorPose3(key, pose_init, prior_noise)

        # Odometry factor from previous keyframe (constant velocity model)
        if idx > 0:
            prev_key = self._pose_key(idx - 1)
            odom = self.poses[idx - 1].between(pose_init)
            odom_noise = gtsam.noiseModel.Isotropic.Sigma(6, self.odom_sigma)
            graph.add(gtsam.BetweenFactorPose3(prev_key, key, odom, odom_noise))

        # Photometric rendering factor (only if we have Gaussians to render)
        if self.gaussian_map.n_gaussians > 0:
            pixel_idx = sample_pixel_indices(self.H, self.W, self.n_pixel_samples, self.rng)
            splat_factor = GaussianSplatFactor(
                gaussian_map=self.gaussian_map,
                target_image=image,
                K=self.K_torch,
                pixel_indices=pixel_idx,
                W=self.W,
                H=self.H,
                device=self.device,
            )
            photo_noise = gtsam.noiseModel.Isotropic.Sigma(
                splat_factor.n_residuals, self.photo_sigma
            )
            graph.add(splat_factor.as_gtsam_factor(key, photo_noise))

        # Add to iSAM2 and iterate for convergence
        initial.insert(key, pose_init)
        self.isam2.update(graph, initial)
        for _ in range(self.tracking_iters - 1):
            self.isam2.update()
        estimate = self.isam2.calculateEstimate()
        optimized_pose = estimate.atPose3(key)
        self.poses[idx] = optimized_pose

        # Initialize new Gaussians from depth
        if depth is not None:
            self._init_gaussians_from_depth(image, depth, optimized_pose)

        # Mapping step: optimize Gaussians with fixed poses
        if self.gaussian_map.n_gaussians > 0 and self.mapping_iters > 0:
            self._mapping_step(image, optimized_pose)

        return pose3_to_matrix(optimized_pose)

    def _init_gaussians_from_depth(self, image, depth, pose):
        """Back-project depth pixels to 3D and add as new Gaussians."""
        fx, fy = self.K_np[0, 0], self.K_np[1, 1]
        cx, cy = self.K_np[0, 2], self.K_np[1, 2]
        T = pose3_to_matrix(pose)

        # Subsample for efficiency
        stride = max(1, min(self.H, self.W) // 32)
        rows = np.arange(0, self.H, stride)
        cols = np.arange(0, self.W, stride)
        rr, cc = np.meshgrid(rows, cols, indexing='ij')
        rr, cc = rr.flatten(), cc.flatten()
        d = depth[rr, cc]
        valid = d > 0
        rr, cc, d = rr[valid], cc[valid], d[valid]

        # Back-project to camera frame
        x_cam = (cc - cx) * d / fx
        y_cam = (rr - cy) * d / fy
        z_cam = d
        pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(d)], axis=1)

        # Transform to world frame
        pts_world = (T @ pts_cam.T).T[:, :3]
        colors_np = image[rr, cc]

        self.gaussian_map.add_gaussians(pts_world, colors_np)

    def _mapping_step(self, image, pose):
        """Optimize Gaussian parameters with the current pose fixed."""
        from gsplat_slam.renderer import render_gaussians

        T = torch.tensor(
            pose3_to_matrix(pose), dtype=torch.float32, device=self.device
        )
        viewmat = torch.inverse(T)
        target = torch.tensor(image, dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam(self.gaussian_map.parameters(), lr=self.mapping_lr)

        for _ in range(self.mapping_iters):
            optimizer.zero_grad()
            rendered, _, _ = render_gaussians(
                means=self.gaussian_map.means,
                quats=self.gaussian_map.quats,
                scales=self.gaussian_map.scales,
                opacities=self.gaussian_map.opacities,
                colors=self.gaussian_map.colors,
                viewmat=viewmat,
                K=self.K_torch,
                W=self.W,
                H=self.H,
            )
            loss = torch.nn.functional.mse_loss(rendered, target)
            loss.backward()
            optimizer.step()

    def add_loop_closure(self, idx_from: int, idx_to: int, relative_pose: np.ndarray, sigma: float = 0.05):
        """Add a loop closure constraint between two keyframes."""
        key_from = self._pose_key(idx_from)
        key_to = self._pose_key(idx_to)
        between = matrix_to_pose3(relative_pose)
        noise = gtsam.noiseModel.Isotropic.Sigma(6, sigma)

        graph = gtsam.NonlinearFactorGraph()
        graph.add(gtsam.BetweenFactorPose3(key_from, key_to, between, noise))

        self.isam2.update(graph, gtsam.Values())
        estimate = self.isam2.calculateEstimate()

        for idx in self.poses:
            self.poses[idx] = estimate.atPose3(self._pose_key(idx))

    def get_all_poses(self) -> dict[int, np.ndarray]:
        """Return all keyframe poses as 4x4 matrices."""
        return {idx: pose3_to_matrix(p) for idx, p in self.poses.items()}
