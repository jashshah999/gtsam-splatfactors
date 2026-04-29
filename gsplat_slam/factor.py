"""GaussianSplatFactor: a GTSAM factor that uses differentiable Gaussian
splatting to compute photometric residuals for pose estimation.

The factor renders the Gaussian map from a candidate camera pose and compares
the rendered image to the observed keyframe at a set of sampled pixels. The
residual is the per-pixel RGB difference; the Jacobian w.r.t. the Pose3 is
computed via torch.autograd through the gsplat rasterizer.
"""

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

from gsplat_slam.pose_utils import gtsam_to_torch_pose


class GaussianSplatFactor:
    """Photometric factor using differentiable Gaussian splatting.

    Evaluates the rendering error between the current Gaussian map rendered
    from a candidate pose and an observed keyframe image, at a fixed set of
    sampled pixel locations.

    Usage with GTSAM:
        factor = GaussianSplatFactor(...)
        gtsam_factor = factor.as_gtsam_factor(pose_key, noise_model)
        graph.add(gtsam_factor)
    """

    def __init__(
        self,
        gaussian_map,
        target_image,      # (H, W, 3) torch tensor, float [0,1]
        K,                 # (3, 3) intrinsics torch tensor
        pixel_indices,     # (n_samples, 2) numpy array of (row, col)
        W: int,
        H: int,
        device: str = "cuda",
    ):
        self.gaussian_map = gaussian_map
        self.device = device
        self.H = H
        self.W = W

        if isinstance(target_image, np.ndarray):
            target_image = torch.tensor(target_image, dtype=torch.float32)
        self.target_image = target_image.to(device)

        if isinstance(K, np.ndarray):
            K = torch.tensor(K, dtype=torch.float32)
        self.K = K.to(device)

        self.pixel_indices = pixel_indices
        self.n_residuals = len(pixel_indices) * 3  # RGB per pixel

    def _render_and_residual(self, viewmat_torch):
        """Render from viewmat and compute residual at sampled pixels.

        Args:
            viewmat_torch: (4, 4) tensor, world-to-camera, requires_grad=True

        Returns:
            residual: (n_residuals,) tensor
        """
        from gsplat_slam.renderer import render_gaussians, compute_photometric_residual

        rendered, _, _ = render_gaussians(
            means=self.gaussian_map.means,
            quats=self.gaussian_map.quats,
            scales=self.gaussian_map.scales,
            opacities=self.gaussian_map.opacities,
            colors=self.gaussian_map.colors,
            viewmat=viewmat_torch,
            K=self.K,
            W=self.W,
            H=self.H,
        )
        return compute_photometric_residual(
            rendered, self.target_image, self.pixel_indices
        )

    def evaluate(self, pose):
        """Evaluate the residual and Jacobian for a gtsam.Pose3.

        The Jacobian is computed via central differences in the Pose3 tangent
        space (Lie algebra).  This avoids having to chain-rule through
        torch.inverse and the Lie group retraction, and is robust to the
        discontinuities in the rasterizer.

        Returns:
            residual: (n_residuals,) numpy array
            jacobian: (n_residuals, 6) numpy array, derivative w.r.t. Pose3 tangent
        """
        residual_np = self._eval_residual_np(pose)
        J_pose = self._numeric_jacobian(pose)
        return residual_np, J_pose

    def _numeric_jacobian(self, pose, eps=1e-5):
        """6-column Jacobian via central differences in the Pose3 tangent space."""
        f0 = self._eval_residual_np(pose)
        n = len(f0)
        J = np.zeros((n, 6))
        for i in range(6):
            delta = np.zeros(6)
            delta[i] = eps
            f_plus = self._eval_residual_np(pose.retract(delta))
            f_minus = self._eval_residual_np(pose.retract(-delta))
            J[:, i] = (f_plus - f_minus) / (2 * eps)
        return J

    def _eval_residual_np(self, pose) -> np.ndarray:
        """Evaluate residual only (no Jacobian), return numpy."""
        with torch.no_grad():
            T_world_cam = gtsam_to_torch_pose(pose, device=self.device)
            viewmat = torch.inverse(T_world_cam)
            residual = self._render_and_residual(viewmat)
            return residual.cpu().numpy()

    def error_at(self, pose) -> float:
        """Squared error (scalar) at a pose. Useful for debugging."""
        r = self._eval_residual_np(pose)
        return 0.5 * np.dot(r, r)

    def as_gtsam_factor(self, pose_key, noise_model=None):
        """Create a gtsam.CustomFactor wrapping this rendering factor.

        Args:
            pose_key: gtsam.Key for the Pose3 variable
            noise_model: gtsam noise model (default: unit isotropic)

        Returns:
            gtsam.CustomFactor
        """
        assert HAS_GTSAM, "gtsam required. pip install gtsam"

        if noise_model is None:
            noise_model = gtsam.noiseModel.Isotropic.Sigma(
                self.n_residuals, 1.0
            )

        def error_func(this, values, jacobians):
            pose = values.atPose3(pose_key)
            residual, J = self.evaluate(pose)
            if jacobians is not None:
                jacobians[0] = J
            return residual

        return gtsam.CustomFactor(
            noise_model, [pose_key], error_func
        )
