"""Keyframe selection and management for incremental 3DGS-SLAM.

Not every frame should be a keyframe. Good keyframe selection:
1. Avoids redundant frames (similar viewpoints waste compute)
2. Ensures sufficient overlap for tracking (don't skip too far)
3. Triggers mapping only when new scene content is visible
"""

import numpy as np
import torch
from typing import Optional


class KeyframeManager:
    """Decides when to create a new keyframe based on pose change and overlap."""

    def __init__(
        self,
        min_translation: float = 0.05,   # 5cm minimum translation
        min_rotation_deg: float = 5.0,    # 5 degree minimum rotation
        max_translation: float = 0.5,     # 50cm max before forcing keyframe
        overlap_threshold: float = 0.6,   # Minimum rendering overlap with last KF
        device: str = "cuda",
    ):
        self.min_translation = min_translation
        self.min_rotation_deg = min_rotation_deg
        self.max_translation = max_translation
        self.overlap_threshold = overlap_threshold
        self.device = device
        self.keyframes: list[dict] = []

    def should_add_keyframe(
        self,
        current_pose: np.ndarray,
        rendered_alpha: Optional[np.ndarray] = None,
    ) -> bool:
        """Check if current frame should become a keyframe.

        Args:
            current_pose: (4, 4) current camera pose
            rendered_alpha: (H, W) alpha map from rendering at current pose
                          (low alpha = new scene content visible)
        """
        if not self.keyframes:
            return True

        last_kf = self.keyframes[-1]
        last_pose = last_kf["pose"]

        # Translation check
        translation = np.linalg.norm(current_pose[:3, 3] - last_pose[:3, 3])
        if translation > self.max_translation:
            return True
        if translation < self.min_translation:
            # Check rotation even if translation is small
            R_rel = last_pose[:3, :3].T @ current_pose[:3, :3]
            angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
            if np.degrees(angle) < self.min_rotation_deg:
                return False

        # Rotation check
        R_rel = last_pose[:3, :3].T @ current_pose[:3, :3]
        angle = np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))
        if np.degrees(angle) > self.min_rotation_deg * 3:
            return True

        # Overlap check (if alpha map available)
        if rendered_alpha is not None:
            overlap = float(np.mean(rendered_alpha > 0.5))
            if overlap < self.overlap_threshold:
                return True

        # Translation above minimum
        if translation > self.min_translation:
            return True

        return False

    def add_keyframe(self, pose: np.ndarray, image: np.ndarray, depth: Optional[np.ndarray] = None):
        """Register a new keyframe."""
        self.keyframes.append({
            "pose": pose.copy(),
            "image": image,
            "depth": depth,
            "idx": len(self.keyframes),
        })

    def get_covisible_keyframes(self, current_pose: np.ndarray, n_max: int = 5) -> list[int]:
        """Get indices of keyframes with most visual overlap to current pose.

        Used for selecting which keyframes to render against during mapping.
        """
        if not self.keyframes:
            return []

        # Score by inverse distance + viewing direction similarity
        scores = []
        curr_pos = current_pose[:3, 3]
        curr_dir = current_pose[:3, 2]  # Z-axis = viewing direction

        for i, kf in enumerate(self.keyframes):
            kf_pos = kf["pose"][:3, 3]
            kf_dir = kf["pose"][:3, 2]

            dist = np.linalg.norm(curr_pos - kf_pos)
            dir_sim = np.dot(curr_dir, kf_dir)  # 1 = same direction, -1 = opposite

            # Prefer nearby keyframes with similar viewing direction
            score = dir_sim / (dist + 0.1)
            scores.append((i, score))

        scores.sort(key=lambda x: -x[1])
        return [idx for idx, _ in scores[:n_max]]

    @property
    def n_keyframes(self) -> int:
        return len(self.keyframes)
