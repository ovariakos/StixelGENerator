#!/usr/bin/env python3
"""Interactive calibration GUI with optional Stixel overlay.

This tool displays a camera image with the corresponding point cloud
projected using adjustable camera matrices. If a StixelWorld (``.stx1``)
file is given it will be visualised on top of the image. Otherwise a
simple Stixel representation is generated from the currently projected
points allowing instant feedback while tuning the matrices.

Use the sliders or press ``i`` to enter numeric values for the intrinsic
and extrinsic parameters. Press ``s`` to save the current matrices to
``calibration.yaml`` inside the dataset directory.
"""

import os
import argparse
import yaml
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import simpledialog
from pathlib import Path
import importlib.util
import sys
import types

from typing import List, Tuple, Dict

WINDOW = "calibration"
DEFAULT_ROOT = Path(__file__).resolve().parents[0]

# --- START OF MODIFICATION ---
# Global dictionary to hold default trackbar values for the reset function
DEFAULT_VALS: Dict[str, int] = {}
# --- END OF MODIFICATION ---

try:
    import stixel  # type: ignore
except Exception:
    stixel = None

try:
    import open3d as o3d  # type: ignore
except Exception:
    o3d = None


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _prepare_libraries(root: Path):
    """Load helper modules for stixel visualisation."""
    sys.path.append(str(root))
    stixel_mod = _load_module("libraries.Stixel", root / "libraries" / "Stixel.py")

    dummy = types.ModuleType("libraries")
    dummy.__path__ = [str(root / "libraries")]
    dummy.StixelClass = stixel_mod.StixelClass
    dummy.Stixel = stixel_mod.Stixel
    sys.modules.setdefault("libraries", dummy)
    sys.modules.setdefault("libraries.Stixel", stixel_mod)

    viz_mod = _load_module("libraries.visualization", root / "libraries" / "visualization.py")

    return stixel_mod, viz_mod


def _to_stixels(
        world: "stixel.StixelWorld", img_size: dict, stixel_mod
) -> Tuple[List[object], int]:
    """Convert a ``stixel.StixelWorld`` to visualization objects."""
    Stixel = stixel_mod.Stixel
    StixelClass = stixel_mod.StixelClass
    point_dtype = stixel_mod.point_dtype
    stixels: List[object] = []
    width = world.stixel[0].width if world.stixel else 8
    for stx in world.stixel:
        top_pt = np.array((0.0, 0.0, 0.0, stx.u, stx.vT, stx.d, stx.label), dtype=point_dtype)
        bot_pt = np.array((0.0, 0.0, 0.0, stx.u, stx.vB, stx.d, stx.label), dtype=point_dtype)
        stixels.append(
            Stixel(top_pt, bot_pt, StixelClass.OBJECT, img_size, grid_step=stx.width)
        )
    return stixels, width


def load_frame(dataset_dir: str, idx: int):
    map_path = os.path.join(dataset_dir, "dataset_map.csv")
    records = pd.read_csv(map_path)
    row = records.iloc[idx]
    img = cv2.imread(os.path.join(dataset_dir, row["image_file"]))
    pts = np.loadtxt(os.path.join(dataset_dir, row["pc_file"]), delimiter=",", skiprows=1)
    return img, pts


def create_trackbars(image_shape):
    global DEFAULT_VALS
    h, w = image_shape[:2]
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, w // 2, h // 2)

    # --- START OF MODIFICATION ---
    # Define and store default values for the reset function
    DEFAULT_VALS = {
        "fx": 1952, "fy": 1104, "cx": w // 2, "cy": h // 2,
        "tx": 500, "ty": 500, "tz": 500,
        "roll": 180, "pitch": 180, "yaw": 180,
    }

    # Intrinsics
    cv2.createTrackbar("fx", WINDOW, DEFAULT_VALS["fx"], 3000, lambda x: None)
    cv2.createTrackbar("fy", WINDOW, DEFAULT_VALS["fy"], 3000, lambda x: None)
    cv2.createTrackbar("cx", WINDOW, DEFAULT_VALS["cx"], w, lambda x: None)
    cv2.createTrackbar("cy", WINDOW, DEFAULT_VALS["cy"], h, lambda x: None)

    # Extrinsic translation
    cv2.createTrackbar("tx", WINDOW, DEFAULT_VALS["tx"], 1000, lambda x: None)
    cv2.createTrackbar("ty", WINDOW, DEFAULT_VALS["ty"], 1000, lambda x: None)
    cv2.createTrackbar("tz", WINDOW, DEFAULT_VALS["tz"], 1000, lambda x: None)

    # Extrinsic rotation
    cv2.createTrackbar("roll", WINDOW, DEFAULT_VALS["roll"], 360, lambda x: None)
    cv2.createTrackbar("pitch", WINDOW, DEFAULT_VALS["pitch"], 360, lambda x: None)
    cv2.createTrackbar("yaw", WINDOW, DEFAULT_VALS["yaw"], 360, lambda x: None)
    # --- END OF MODIFICATION ---


def get_values(image_shape):
    fx = cv2.getTrackbarPos("fx", WINDOW)
    fy = cv2.getTrackbarPos("fy", WINDOW)
    cx = cv2.getTrackbarPos("cx", WINDOW)
    cy = cv2.getTrackbarPos("cy", WINDOW)

    tx = (cv2.getTrackbarPos("tx", WINDOW) - 500) / 100.0
    ty = (cv2.getTrackbarPos("ty", WINDOW) - 500) / 100.0
    tz = (cv2.getTrackbarPos("tz", WINDOW) - 500) / 100.0

    roll = np.deg2rad(cv2.getTrackbarPos("roll", WINDOW) - 180)
    pitch = np.deg2rad(cv2.getTrackbarPos("pitch", WINDOW) - 180)
    yaw = np.deg2rad(cv2.getTrackbarPos("yaw", WINDOW) - 180)

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )
    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    Rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    R = Rz @ Ry @ Rx

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]

    P = np.hstack((K, np.zeros((3, 1))))
    return K, P, T


def project_points(pts, K, T):
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    cam = T.dot(pts_h.T)
    proj = K.dot(cam[:3, :])
    proj[:2] /= proj[2, :]
    return proj[:2].T, proj[2]


def overlay_points(img, pts_uv, depth):
    overlay = img.copy()
    h, w = img.shape[:2]
    for (u, v), d in zip(pts_uv, depth):
        if d <= 0 or not np.isfinite(u) or not np.isfinite(v):
            continue
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < w and 0 <= vi < h:
            cv2.circle(overlay, (ui, vi), 2, (0, 255, 0), -1)
    return overlay


def overlay_stixels(img, stixels, viz_mod, width):
    if not stixels:
        return img
    result = viz_mod.draw_stixels_on_image(img.copy(), stixels, stixel_width=width, draw_grid=False)
    return np.array(result)


def stixels_from_projection(uv, depth, img_shape, stixel_mod, width=8):
    """Create simple stixels from projected points."""
    w, h = img_shape[1], img_shape[0]
    Stixel = stixel_mod.Stixel
    StixelClass = stixel_mod.StixelClass
    point_dtype = stixel_mod.point_dtype
    stixels: List[object] = []
    u = uv[:, 0]
    v = uv[:, 1]
    valid = (depth > 0) & np.isfinite(u) & np.isfinite(v)
    u = u[valid]
    v = v[valid]
    d = depth[valid]
    for col in range(0, w, width):
        mask = (u >= col) & (u < col + width) & (v >= 0) & (v < h)
        if np.sum(mask) == 0:
            continue
        top = float(np.min(v[mask]))
        bottom = float(np.max(v[mask]))
        depth_med = float(np.median(d[mask]))
        top_pt = np.array((0.0, 0.0, 0.0, col, top, depth_med, 0), dtype=point_dtype)
        bot_pt = np.array((0.0, 0.0, 0.0, col, bottom, depth_med, 0), dtype=point_dtype)
        stixels.append(Stixel(top_pt, bot_pt, StixelClass.OBJECT, {"width": w, "height": h}, grid_step=width))
    return stixels


def _create_lidar_window(pts):
    """Create a 3D viewer for the lidar point cloud with enhanced visibility usable alongside OpenCV windows."""
    if o3d is None:
        return None, None, None

    # Center point cloud at origin for better camera manipulation
    pts_np = np.asarray(pts)
    center = pts_np.mean(axis=0)
    pts_centered = pts_np - center

    # Initialize visualizer (non-blocking render loop)
    vis = o3d.visualization.Visualizer()
    vis.create_window("lidar", width=800, height=600)

    # Render options: dark background, larger points
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.0, 0.0, 0.0])
    opt.point_size = 5.0

    # Point cloud geometry
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_centered)

    # --- START OF MODIFICATION ---
    # Implement a more sensitive, diverging colormap with outlier filtering
    z_coords = pts_np[:, 2]

    # 1. Filter outliers by defining a robust range using percentiles
    p_low, p_high = 1.0, 99.0  # Use 1st and 99th percentiles
    z_min_robust = np.percentile(z_coords, p_low)
    z_max_robust = np.percentile(z_coords, p_high)

    # 2. Calculate the mean of the inlier points to use as the center
    inlier_mask = (z_coords > z_min_robust) & (z_coords < z_max_robust)
    z_mean = np.mean(z_coords[inlier_mask])

    # 3. Define the three colors for the diverging map
    color_low = np.array([0.2, 0.6, 1.0])  # Light Blue
    color_mid = np.array([0.9, 0.9, 0.9])  # Off-White (for the mean)
    color_high = np.array([1.0, 0.0, 0.0])  # Red

    # 4. Create an empty color array and process the two halves of the map
    colors = np.zeros((len(z_coords), 3))

    # Create masks for points below and above the mean
    lower_mask = z_coords <= z_mean
    upper_mask = z_coords > z_mean

    # Normalize and interpolate for the lower half (blue -> white)
    range_low = z_mean - z_min_robust
    if range_low > 0:
        # Clip values to the robust range before normalization
        lower_z_clipped = np.clip(z_coords[lower_mask], z_min_robust, z_mean)
        norm_low = (lower_z_clipped - z_min_robust) / range_low
        colors[lower_mask] = (1 - norm_low)[:, np.newaxis] * color_low + norm_low[:, np.newaxis] * color_mid
    else:  # If all points are the same, color them mid
        colors[lower_mask] = color_mid

    # Normalize and interpolate for the upper half (white -> red)
    range_high = z_max_robust - z_mean
    if range_high > 0:
        # Clip values to the robust range before normalization
        upper_z_clipped = np.clip(z_coords[upper_mask], z_mean, z_max_robust)
        norm_high = (upper_z_clipped - z_mean) / range_high
        colors[upper_mask] = (1 - norm_high)[:, np.newaxis] * color_mid + norm_high[:, np.newaxis] * color_high
    else:  # If all points are the same, color them mid
        colors[upper_mask] = color_mid

    pcd.colors = o3d.utility.Vector3dVector(colors)
    # --- END OF MODIFICATION ---

    vis.add_geometry(pcd)

    # Reference axes
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    vis.add_geometry(coord_frame)

    # Frustum geometry
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(np.zeros((5, 3)))
    frustum.lines = o3d.utility.Vector2iVector([
        [0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]
    ])
    frustum.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]] * 8)
    vis.add_geometry(frustum)

    # Initial render so window shows content
    vis.poll_events()
    vis.update_renderer()

    # Setup camera towards origin
    ctr = vis.get_view_control()
    ctr.set_lookat([0.0, 0.0, 0.0])
    ctr.set_up([0.0, -1.0, 0.0])
    ctr.set_front([0.0, 0.0, -1.0])
    ctr.set_zoom(0.7)

    return vis, pcd, frustum


def _update_frustum(lineset, K, T, img_shape, far=5.0):
    """Update frustum line set to match the current camera pose."""
    if lineset is None:
        return
    w, h = img_shape[1], img_shape[0]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    T_inv = np.linalg.inv(T)

    def pixel_to_cam(u, v, depth):
        x = (u - cx) / fx * depth
        y = (v - cy) / fy * depth
        return np.array([x, y, depth, 1.0])

    corners = [
        pixel_to_cam(0, 0, far),
        pixel_to_cam(w, 0, far),
        pixel_to_cam(w, h, far),
        pixel_to_cam(0, h, far),
    ]
    origin = np.array([0.0, 0.0, 0.0, 1.0])
    pts = [origin] + corners
    pts_world = [(T_inv @ p)[:3] for p in pts]
    lineset.points = o3d.utility.Vector3dVector(np.array(pts_world))


# --- START OF MODIFICATION ---
def reset_sliders_to_default():
    """Reset all trackbars to their initial default values."""
    print("Resetting sliders to default values.")
    for name, value in DEFAULT_VALS.items():
        cv2.setTrackbarPos(name, WINDOW, value)


def prompt_single_value():
    """Ask the user for a single parameter and its new value."""
    root = tk.Tk()
    root.withdraw()

    # Map parameter names to their trackbar scaling factors
    param_map = {
        "fx": {"scale": 1, "offset": 0}, "fy": {"scale": 1, "offset": 0},
        "cx": {"scale": 1, "offset": 0}, "cy": {"scale": 1, "offset": 0},
        "tx": {"scale": 100, "offset": 500}, "ty": {"scale": 100, "offset": 500},
        "tz": {"scale": 100, "offset": 500},
        "roll": {"scale": 1, "offset": 180},
        "pitch": {"scale": 1, "offset": 180},
        "yaw": {"scale": 1, "offset": 180}
    }

    name = simpledialog.askstring("Input", f"Enter parameter name ({', '.join(param_map.keys())}):")
    if not name or name not in param_map:
        if name: print(f"Invalid parameter name: {name}")
        root.destroy()
        return

    val_str = simpledialog.askstring("Input", f"Enter new value for '{name}':")
    try:
        val = float(val_str)
    except (ValueError, TypeError):
        if val_str: print(f"Invalid numeric value: {val_str}")
        root.destroy()
        return

    # Convert user-friendly value to the integer trackbar position
    # Example for tx: user enters -0.1 -> trackbar pos is -0.1 * 100 + 500 = 490
    # Example for roll: user enters 90 -> trackbar pos is 90 + 180 = 270
    p = param_map[name]
    trackbar_val = int(round(val * p["scale"] + p["offset"]))

    cv2.setTrackbarPos(name, WINDOW, trackbar_val)
    print(f"Set {name} to {val} (Trackbar: {trackbar_val})")
    root.destroy()


# --- END OF MODIFICATION ---


def main():
    parser = argparse.ArgumentParser(description="Manual calibration tool with Stixel overlay")
    parser.add_argument("--data", required=True, help="Dataset directory")
    parser.add_argument("--index", type=int, default=0, help="Frame index")
    parser.add_argument("--stx", help="Optional StixelWorld .stx1 file")
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help="Repository root containing the 'libraries' directory",
    )
    args = parser.parse_args()

    img, pts = load_frame(args.data, args.index)

    root = Path(args.root).resolve()
    stixel_mod, viz_mod = _prepare_libraries(root)

    vis, pcd, frustum = _create_lidar_window(pts)

    stixels = []
    width = 8
    stx_file_mode = False
    if args.stx and stixel is not None:
        stx_path = Path(args.stx)
        if not stx_path.is_absolute():
            stx_path = Path(args.data) / args.stx
        world = stixel.read(str(stx_path))
        img_size = {"width": img.shape[1], "height": img.shape[0]}
        stixels, width = _to_stixels(world, img_size, stixel_mod)
        stx_file_mode = True

    create_trackbars(img.shape)

    # --- START OF MODIFICATION ---
    print("Press 'i' for manual input, 'r' to reset, 's' to save, 'q' to quit.")
    while True:
        K, P, T = get_values(img.shape)
        uv, depth = project_points(pts, K, T)
        disp = overlay_points(img, uv, depth)
        if stx_file_mode:
            display_stixels = stixels
        else:
            display_stixels = stixels_from_projection(uv, depth, img.shape, stixel_mod, width)
        if display_stixels:
            disp = overlay_stixels(disp, display_stixels, viz_mod, width)

        cv2.putText(disp, "i: input, r: reset, s: save, q: quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.imshow(WINDOW, disp)

        _update_frustum(frustum, K, T, img.shape)
        if vis is not None:
            vis.update_geometry(frustum)
            vis.poll_events()
            vis.update_renderer()

        key = cv2.waitKey(10) & 0xFF
        if key == ord("i"):
            prompt_single_value()
        elif key == ord("r"):
            reset_sliders_to_default()
        # --- END OF MODIFICATION ---
        elif key == ord("s"):
            calib = {"K": K.tolist(), "P": P.tolist(), "R": np.eye(4).tolist(), "T": T.tolist()}
            with open(os.path.join(args.data, "calibration.yaml"), "w") as f:
                yaml.dump(calib, f)
            print("Saved calibration.yaml")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    if vis is not None:
        vis.destroy_window()


if __name__ == "__main__":
    main()