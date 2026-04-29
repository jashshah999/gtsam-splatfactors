"""Gaussian map: stores and optimizes the 3D Gaussian parameters."""

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError("PyTorch is required. pip install torch")


class GaussianMap(nn.Module):
    """A collection of 3D Gaussians that can be rendered and optimized."""

    def __init__(self, n_gaussians: int = 0, device: str = "cuda"):
        super().__init__()
        self.device = device
        if n_gaussians > 0:
            self._init_random(n_gaussians)

    def _init_random(self, n: int):
        self.means = nn.Parameter(torch.randn(n, 3, device=self.device) * 0.5)
        self.quats = nn.Parameter(torch.randn(n, 4, device=self.device))
        self.quats.data = torch.nn.functional.normalize(self.quats.data, dim=-1)
        self.scales = nn.Parameter(torch.full((n, 3), -3.0, device=self.device))
        self.opacities = nn.Parameter(torch.full((n,), 2.0, device=self.device))
        self.colors = nn.Parameter(torch.rand(n, 3, device=self.device))

    @classmethod
    def from_pointcloud(cls, points, colors=None, device="cuda"):
        """Initialize Gaussians from a point cloud.

        Args:
            points: (N, 3) numpy array or torch tensor of 3D points.
            colors: (N, 3) optional RGB values in [0, 1].
        """
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        n = len(points)
        gmap = cls(n_gaussians=0, device=device)

        gmap.means = nn.Parameter(points.to(device))
        gmap.quats = nn.Parameter(
            torch.nn.functional.normalize(
                torch.randn(n, 4, device=device), dim=-1
            )
        )
        gmap.scales = nn.Parameter(torch.full((n, 3), -3.0, device=device))
        gmap.opacities = nn.Parameter(torch.full((n,), 2.0, device=device))

        if colors is not None:
            if not isinstance(colors, torch.Tensor):
                colors = torch.tensor(colors, dtype=torch.float32)
            gmap.colors = nn.Parameter(colors.to(device))
        else:
            gmap.colors = nn.Parameter(torch.full((n, 3), 0.5, device=device))

        return gmap

    def add_gaussians(self, means, colors=None):
        """Add new Gaussians to the map (e.g. from new depth observations)."""
        if not isinstance(means, torch.Tensor):
            means = torch.tensor(means, dtype=torch.float32, device=self.device)
        n_new = len(means)

        new_quats = torch.nn.functional.normalize(
            torch.randn(n_new, 4, device=self.device), dim=-1
        )
        new_scales = torch.full((n_new, 3), -3.0, device=self.device)
        new_opacities = torch.full((n_new,), 2.0, device=self.device)
        if colors is not None:
            if not isinstance(colors, torch.Tensor):
                colors = torch.tensor(colors, dtype=torch.float32, device=self.device)
            new_colors = colors
        else:
            new_colors = torch.full((n_new, 3), 0.5, device=self.device)

        self.means = nn.Parameter(torch.cat([self.means.data, means]))
        self.quats = nn.Parameter(torch.cat([self.quats.data, new_quats]))
        self.scales = nn.Parameter(torch.cat([self.scales.data, new_scales]))
        self.opacities = nn.Parameter(torch.cat([self.opacities.data, new_opacities]))
        self.colors = nn.Parameter(torch.cat([self.colors.data, new_colors]))

    @property
    def n_gaussians(self) -> int:
        return len(self.means)
