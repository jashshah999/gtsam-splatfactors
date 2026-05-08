"""Automatic loop closure detection for 3DGS-SLAM.

Combines rendering-based overlap detection with appearance matching.
When a loop is detected, it estimates the relative pose via photometric
alignment against the Gaussian map rendered from the candidate pose.
"""

import numpy as np
import torch
from typing import Optional


class LoopDetector:
    """Detects revisited locations and estimates loop closure poses."""

    def __init__(
        self,
        min_frame_gap: int = 10,
        similarity_threshold: float = 0.7,
        max_candidates: int = 5,
        device: str = "cuda",
    ):
        self.min_frame_gap = min_frame_gap
        self.similarity_threshold = similarity_threshold
        self.max_candidates = max_candidates
        self.device = device
        self.descriptors: list[np.ndarray] = []
        self.frame_indices: list[int] = []
        self._model = None

    def add_frame(self, image: np.ndarray, frame_idx: int):
        """Add a frame's descriptor to the database."""
        desc = self._extract_descriptor(image)
        self.descriptors.append(desc)
        self.frame_indices.append(frame_idx)

    def detect(self, image: np.ndarray, current_idx: int) -> list[tuple[int, float]]:
        """Check if current frame revisits a previous location.

        Returns list of (frame_idx, similarity_score) for detected loop closures.
        """
        if len(self.descriptors) < self.min_frame_gap:
            return []

        query_desc = self._extract_descriptor(image)

        # Compare against all previous frames (excluding recent ones)
        candidates = []
        for i, (desc, idx) in enumerate(zip(self.descriptors, self.frame_indices)):
            if current_idx - idx < self.min_frame_gap:
                continue
            sim = float(np.dot(query_desc, desc))
            if sim > self.similarity_threshold:
                candidates.append((idx, sim))

        candidates.sort(key=lambda x: -x[1])
        return candidates[:self.max_candidates]

    def _extract_descriptor(self, image: np.ndarray) -> np.ndarray:
        """Extract a global descriptor using DINOv2."""
        if self._model is None:
            self._model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", pretrained=True
            ).to(self.device).eval()

        import cv2
        from torchvision import transforms

        img = cv2.resize(image, (224, 224))
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        tensor = torch.tensor(img, device=self.device, dtype=torch.float32)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = transform(tensor)

        with torch.no_grad():
            feat = self._model(tensor)
        feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat.squeeze().cpu().numpy()

    def cleanup(self):
        """Free GPU memory from the descriptor model."""
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()


def estimate_loop_relative_pose(
    gaussian_map,
    image_current: np.ndarray,
    pose_current: np.ndarray,
    pose_candidate: np.ndarray,
    K: np.ndarray,
    W: int, H: int,
    device: str = "cuda",
    n_iters: int = 20,
) -> tuple[np.ndarray, float]:
    """Estimate relative pose for loop closure via photometric alignment.

    Renders the Gaussian map from the candidate pose and refines it
    to match the current frame. Returns the refined relative pose
    and alignment confidence (lower loss = higher confidence).
    """
    from gsplat_slam.renderer import render_gaussians
    from gsplat_slam.pose_utils import matrix_to_pose3, pose3_to_matrix

    K_torch = torch.tensor(K, dtype=torch.float32, device=device)
    target = torch.tensor(image_current, dtype=torch.float32, device=device)

    # Start from candidate pose and refine
    pose_t = torch.tensor(pose_candidate, dtype=torch.float32, device=device)
    delta = torch.zeros(6, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=1e-3)

    best_loss = float('inf')
    for _ in range(n_iters):
        optimizer.zero_grad()

        # Apply delta to candidate pose
        dR = _so3_exp(delta[:3])
        dt = delta[3:]
        viewmat = torch.inverse(pose_t.clone())
        viewmat[:3, :3] = viewmat[:3, :3] @ dR
        viewmat[:3, 3] = viewmat[:3, 3] + dt

        rendered, _, _ = render_gaussians(
            means=gaussian_map.means,
            quats=gaussian_map.quats,
            scales=gaussian_map.scales,
            opacities=gaussian_map.opacities,
            colors=gaussian_map.colors,
            viewmat=viewmat,
            K=K_torch,
            W=W, H=H,
        )

        loss = torch.nn.functional.l1_loss(rendered, target)
        loss.backward()
        optimizer.step()
        best_loss = min(best_loss, loss.item())

    # Compute final relative pose
    with torch.no_grad():
        dR = _so3_exp(delta[:3])
        dt = delta[3:]
        refined_viewmat = torch.inverse(pose_t)
        refined_viewmat[:3, :3] = refined_viewmat[:3, :3] @ dR
        refined_viewmat[:3, 3] = refined_viewmat[:3, 3] + dt
        refined_c2w = torch.inverse(refined_viewmat)

    relative_pose = np.linalg.inv(pose_current) @ refined_c2w.cpu().numpy()
    confidence = max(0, 1.0 - best_loss * 5)

    return relative_pose, confidence


def _so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """Exponential map so(3) -> SO(3)."""
    theta = torch.norm(omega)
    if theta < 1e-6:
        return torch.eye(3, device=omega.device, dtype=omega.dtype)
    k = omega / theta
    K = torch.tensor([
        [0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]
    ], device=omega.device, dtype=omega.dtype)
    return torch.eye(3, device=omega.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * K @ K
