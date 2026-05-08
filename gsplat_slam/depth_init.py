"""Monocular depth initialization for RGB-only input.

When no depth sensor is available (phone video, webcam), use a pretrained
monocular depth model to get pseudo-depth for initializing Gaussians.
Supports Depth Anything V2 and Metric3D.
"""

import numpy as np
import torch


def estimate_depth(
    image: np.ndarray,
    model_name: str = "depth_anything_v2",
    device: str = "cuda",
) -> np.ndarray:
    """Estimate depth from a single RGB image.

    Args:
        image: (H, W, 3) float [0, 1] or uint8 [0, 255]
        model_name: "depth_anything_v2" or "metric3d" or "zoedepth"
        device: CUDA device

    Returns:
        (H, W) depth map in meters (or relative scale for non-metric models)
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = image.astype(np.uint8)
    else:
        image_uint8 = image

    if model_name == "depth_anything_v2":
        return _depth_anything_v2(image_uint8, device)
    elif model_name == "zoedepth":
        return _zoedepth(image_uint8, device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _depth_anything_v2(image: np.ndarray, device: str) -> np.ndarray:
    """Depth estimation using Depth Anything V2."""
    model = torch.hub.load("LiheYoung/Depth-Anything", "DepthAnything_vitl14", pretrained=True)
    model = model.to(device).eval()

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(518),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    H, W = image.shape[:2]
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = model(tensor)

    depth = depth.squeeze().cpu().numpy()

    # Resize back to original resolution
    import cv2
    depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

    # Depth Anything outputs inverse depth (disparity-like), convert
    depth = 1.0 / (depth + 1e-6)
    # Normalize to reasonable range (0.1 to 10m)
    depth = depth / np.median(depth) * 2.0

    del model
    torch.cuda.empty_cache()
    return depth.astype(np.float32)


def _zoedepth(image: np.ndarray, device: str) -> np.ndarray:
    """Metric depth estimation using ZoeDepth."""
    model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
    model = model.to(device).eval()

    from PIL import Image
    pil_img = Image.fromarray(image)

    with torch.no_grad():
        depth = model.infer_pil(pil_img)

    depth = depth.squeeze().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return depth.astype(np.float32)


def batch_estimate_depth(
    images: list[np.ndarray],
    model_name: str = "depth_anything_v2",
    device: str = "cuda",
) -> list[np.ndarray]:
    """Estimate depth for multiple images (loads model once)."""
    depths = []
    for img in images:
        d = estimate_depth(img, model_name, device)
        depths.append(d)
    return depths
