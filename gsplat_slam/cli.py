"""CLI for gtsam-splatfactors."""

import click


@click.group()
@click.version_option(version="0.2.0")
def main():
    """gtsam-splatfactors: 3DGS-SLAM with factor graph backend."""
    pass


@main.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("-o", "--output", default="output/slam", help="Output directory")
@click.option("--max-frames", type=int, default=200)
@click.option("--fps", type=float, default=5.0, help="Target FPS for frame extraction")
@click.option("--mapping-iters", type=int, default=50, help="Gaussian optimization iters per keyframe")
@click.option("--no-loop-closure", is_flag=True, help="Disable loop closure detection")
@click.option("--visualize", is_flag=True, help="Launch viser viewer")
@click.option("--device", default="cuda")
def run(video_path, output, max_frames, fps, mapping_iters, no_loop_closure, visualize, device):
    """Run 3DGS-SLAM on a video file.

    Extracts frames, estimates monocular depth, runs incremental SLAM
    with factor graph backend, and exports to COLMAP/nerfstudio/PLY.

    Example:
        gsplat-slam run my_video.mp4 -o output/scene
    """
    from .video_runner import run_video

    run_video(
        video_path=video_path,
        output_dir=output,
        max_frames=max_frames,
        target_fps=fps,
        n_mapping_iters=mapping_iters,
        enable_loop_closure=not no_loop_closure,
        enable_visualization=visualize,
        device=device,
    )


@main.command()
@click.argument("image_dir", type=click.Path(exists=True))
@click.option("-o", "--output", default="output/slam")
@click.option("--mapping-iters", type=int, default=50)
@click.option("--no-loop-closure", is_flag=True)
@click.option("--device", default="cuda")
def images(image_dir, output, mapping_iters, no_loop_closure, device):
    """Run 3DGS-SLAM on a directory of images.

    Example:
        gsplat-slam images path/to/frames/ -o output/scene
    """
    import os
    import numpy as np
    import cv2
    from pathlib import Path
    from .slam import SplatSLAM
    from .depth_init import estimate_depth
    from .keyframe_manager import KeyframeManager
    from .exporters import export_all

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted([str(p) for p in Path(image_dir).iterdir() if p.suffix.lower() in exts])
    if not paths:
        click.echo("No images found.")
        return

    img0 = cv2.imread(paths[0])
    H, W = img0.shape[:2]
    fx = W / (2 * np.tan(np.radians(30)))
    K = np.array([[fx, 0, W/2], [0, fx, H/2], [0, 0, 1]], dtype=np.float64)

    slam = SplatSLAM(K=K, W=W, H=H, device=device, mapping_iters=mapping_iters)

    for i, path in enumerate(paths):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        depth = estimate_depth(img, device=device)
        pose = slam.add_keyframe(img, depth)
        if i % 10 == 0:
            click.echo(f"  Frame {i}/{len(paths)}, {slam.gaussian_map.n_gaussians} Gaussians")

    os.makedirs(output, exist_ok=True)
    export_all(slam, output, paths)
    click.echo(f"\nDone. Output: {output}/")


@main.command()
def check():
    """Check if all dependencies are installed."""
    import sys
    click.echo("gtsam-splatfactors dependency check")
    click.echo("=" * 40)

    deps = {
        "torch": "torch",
        "gsplat": "gsplat",
        "gtsam": "gtsam",
        "cv2": "opencv-python",
        "viser": "viser",
        "scipy": "scipy",
    }

    all_ok = True
    for name, pkg in deps.items():
        try:
            __import__(name)
            click.echo(f"  {pkg}: OK")
        except ImportError:
            click.echo(f"  {pkg}: MISSING (pip install {pkg})")
            all_ok = False

    if all_ok:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            click.echo(f"\n  GPU: {gpu} ({vram:.1f} GB)")
        else:
            click.echo("\n  GPU: None (CPU only, will be slow)")


if __name__ == "__main__":
    main()
