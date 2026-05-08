"""One-command video → 3DGS-SLAM pipeline.

Takes a video file and runs the full SLAM pipeline:
1. Extract frames with blur filtering
2. Estimate monocular depth
3. Run SplatSLAM with keyframe selection + loop closure
4. Export to COLMAP / nerfstudio / PLY
"""

import os
import time
import json
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Optional

from .slam import SplatSLAM
from .keyframe_manager import KeyframeManager
from .loop_detector import LoopDetector
from .depth_init import estimate_depth
from .exporters import export_all


def run_video(
    video_path: str,
    output_dir: str,
    max_frames: int = 200,
    target_fps: float = 5.0,
    min_blur: float = 50.0,
    depth_model: str = "depth_anything_v2",
    n_mapping_iters: int = 50,
    enable_loop_closure: bool = True,
    enable_visualization: bool = False,
    device: str = "cuda",
) -> dict:
    """Run full 3DGS-SLAM pipeline on a video file.

    Args:
        video_path: Path to input video
        output_dir: Output directory
        max_frames: Maximum frames to process
        target_fps: Target frame rate for extraction
        min_blur: Minimum Laplacian variance (filters blurry frames)
        depth_model: Monocular depth model
        n_mapping_iters: Gaussian optimization iterations per keyframe
        enable_loop_closure: Enable DINOv2 loop closure detection
        enable_visualization: Launch viser viewer
        device: CUDA device

    Returns:
        dict with timing, metrics, and output paths
    """
    t_start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Step 1: Extract frames
    print("Step 1: Extracting frames...")
    image_paths, images = _extract_frames(
        video_path, str(frames_dir), max_frames, target_fps, min_blur
    )
    N = len(images)
    print(f"  {N} frames extracted")

    if N < 3:
        raise ValueError(f"Only {N} usable frames. Need at least 3.")

    H, W = images[0].shape[:2]
    # Estimate intrinsics (assume ~60 degree FOV)
    fx = W / (2 * np.tan(np.radians(30)))
    K = np.array([[fx, 0, W/2], [0, fx, H/2], [0, 0, 1]], dtype=np.float64)

    # Step 2: Estimate monocular depth
    print("Step 2: Estimating monocular depth...")
    t_depth = time.time()
    depths = []
    for i, img in enumerate(images):
        if i % 10 == 0:
            print(f"  Depth {i}/{N}...")
        depth = estimate_depth(img, model_name=depth_model, device=device)
        depths.append(depth)
    t_depth = time.time() - t_depth
    print(f"  Depth estimation: {t_depth:.1f}s")

    # Step 3: Run SLAM
    print("Step 3: Running 3DGS-SLAM...")
    t_slam = time.time()

    slam = SplatSLAM(
        K=K, W=W, H=H,
        device=device,
        mapping_iters=n_mapping_iters,
        n_pixel_samples=1024,
    )

    kf_manager = KeyframeManager(
        min_translation=0.03,
        min_rotation_deg=3.0,
        device=device,
    )

    loop_detector = LoopDetector(device=device) if enable_loop_closure else None

    vis = None
    if enable_visualization:
        from .visualization import SLAMVisualizer
        vis = SLAMVisualizer()

    prev_pose = np.eye(4)
    for i in range(N):
        image = images[i]
        depth = depths[i]

        # Check if this should be a keyframe
        if i > 0 and not kf_manager.should_add_keyframe(prev_pose):
            continue

        # Add keyframe to SLAM
        pose = slam.add_keyframe(image, depth, init_pose=prev_pose)
        prev_pose = pose
        kf_manager.add_keyframe(pose, image, depth)

        # Loop closure detection
        if loop_detector is not None and kf_manager.n_keyframes > 10:
            loops = loop_detector.detect(image, kf_manager.n_keyframes - 1)
            for loop_idx, score in loops:
                # Get relative pose via photometric alignment
                from .loop_detector import estimate_loop_relative_pose
                loop_pose = list(slam.get_all_poses().values())[loop_idx]
                rel_pose, confidence = estimate_loop_relative_pose(
                    slam.gaussian_map, image, pose, loop_pose, K, W, H, device
                )
                if confidence > 0.3:
                    slam.add_loop_closure(loop_idx, kf_manager.n_keyframes - 1, rel_pose)
                    print(f"  Loop closure: {loop_idx} ↔ {kf_manager.n_keyframes - 1} (conf={confidence:.2f})")

        # Add frame descriptor for loop closure
        if loop_detector is not None:
            loop_detector.add_frame(image, kf_manager.n_keyframes - 1)

        # Visualization
        if vis is not None:
            vis.update_pose(pose, kf_manager.n_keyframes - 1)
            if kf_manager.n_keyframes % 5 == 0:
                vis.update_map(slam.gaussian_map)
                vis.update_trajectory(slam.get_all_poses())

        if i % 20 == 0:
            print(f"  Frame {i}/{N}, {kf_manager.n_keyframes} keyframes, {slam.gaussian_map.n_gaussians} Gaussians")

    t_slam = time.time() - t_slam

    # Cleanup loop detector
    if loop_detector:
        loop_detector.cleanup()

    # Step 4: Export
    print("Step 4: Exporting...")
    export_all(slam, str(output_dir), image_paths)

    t_total = time.time() - t_start

    summary = {
        "input": video_path,
        "n_frames": N,
        "n_keyframes": kf_manager.n_keyframes,
        "n_gaussians": slam.gaussian_map.n_gaussians,
        "timing": {
            "depth_estimation_s": round(t_depth, 1),
            "slam_s": round(t_slam, 1),
            "total_s": round(t_total, 1),
        },
        "output_dir": str(output_dir),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {t_total:.0f}s")
    print(f"  Keyframes: {kf_manager.n_keyframes}")
    print(f"  Gaussians: {slam.gaussian_map.n_gaussians}")
    print(f"  Output: {output_dir}/")

    return summary


def _extract_frames(video_path, output_dir, max_frames, target_fps, min_blur):
    """Extract frames from video with blur filtering."""
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Compute frame skip
    skip = max(1, int(video_fps / target_fps))

    images = []
    paths = []
    frame_idx = 0

    while cap.isOpened() and len(images) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        # Blur check
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < min_blur:
            frame_idx += 1
            continue

        # Save and store
        path = os.path.join(output_dir, f"frame_{len(images):04d}.jpg")
        cv2.imwrite(path, frame)
        paths.append(path)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        images.append(frame_rgb)
        frame_idx += 1

    cap.release()
    return paths, images
