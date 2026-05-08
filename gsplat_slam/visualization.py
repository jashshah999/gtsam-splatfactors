"""Real-time visualization using viser for 3DGS-SLAM.

Shows the Gaussian map being built incrementally with camera frustums,
trajectory, and loop closure connections.
"""

import numpy as np
from typing import Optional

try:
    import viser
    HAS_VISER = True
except ImportError:
    HAS_VISER = False


class SLAMVisualizer:
    """Live visualization of the SLAM reconstruction."""

    def __init__(self, port: int = 8080):
        assert HAS_VISER, "viser required: pip install viser"
        self.server = viser.ViserServer(port=port)
        self.trajectory_points = []
        self.n_frames = 0

    def update_map(self, gaussian_map, subsample: int = 10):
        """Update the point cloud display with current Gaussian means."""
        import torch
        means = gaussian_map.means.detach().cpu().numpy()
        colors = gaussian_map.colors.detach().cpu().numpy()
        colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

        # Subsample for performance
        if len(means) > 50000:
            idx = np.random.choice(len(means), 50000, replace=False)
            means = means[idx]
            colors = colors[idx]

        self.server.scene.add_point_cloud(
            "map", points=means, colors=colors, point_size=0.003
        )

    def update_pose(self, pose_c2w: np.ndarray, frame_idx: int, color: tuple = (0, 200, 255)):
        """Add a camera frustum at the given pose."""
        position = pose_c2w[:3, 3]
        self.trajectory_points.append(position)
        self.n_frames = frame_idx + 1

        # Draw camera frustum
        R = pose_c2w[:3, :3]
        size = 0.05
        corners = np.array([
            [-size, -size, size * 1.5],
            [size, -size, size * 1.5],
            [size, size, size * 1.5],
            [-size, size, size * 1.5],
        ])
        corners_world = (R @ corners.T).T + position

        # Draw frustum edges
        for i in range(4):
            self.server.scene.add_line_segments(
                f"frustum_{frame_idx}_{i}",
                points=np.array([position, corners_world[i]]),
                colors=np.array([color, color], dtype=np.uint8),
                line_width=1.0,
            )

    def update_trajectory(self, poses: dict, color: tuple = (0, 200, 255)):
        """Draw the full trajectory as a line."""
        if len(poses) < 2:
            return

        sorted_idx = sorted(poses.keys())
        points = np.array([poses[i][:3, 3] for i in sorted_idx])
        colors = np.tile(np.array(color, dtype=np.uint8), (len(points), 1))

        self.server.scene.add_line_segments(
            "trajectory",
            points=points,
            colors=colors,
            line_width=2.0,
        )

    def add_loop_closure_line(self, pose_i: np.ndarray, pose_j: np.ndarray, idx: int):
        """Draw a line showing a loop closure connection."""
        points = np.array([pose_i[:3, 3], pose_j[:3, 3]])
        colors = np.array([[255, 0, 0], [255, 0, 0]], dtype=np.uint8)
        self.server.scene.add_line_segments(
            f"loop_{idx}",
            points=points,
            colors=colors,
            line_width=3.0,
        )

    def add_ground_truth_trajectory(self, gt_poses: np.ndarray, color: tuple = (0, 255, 0)):
        """Overlay ground truth trajectory for comparison."""
        points = gt_poses[:, :3, 3]
        colors = np.tile(np.array(color, dtype=np.uint8), (len(points), 1))
        self.server.scene.add_line_segments(
            "gt_trajectory",
            points=points,
            colors=colors,
            line_width=2.0,
        )
