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

from typing import List, Tuple

WINDOW = "calibration"
DEFAULT_ROOT = Path(__file__).resolve().parents[0]

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
    h, w = image_shape[:2]
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    # Intrinsics
    cv2.createTrackbar("fx", WINDOW, 1952, 3000, lambda x: None)
    cv2.createTrackbar("fy", WINDOW, 1104, 3000, lambda x: None)
    cv2.createTrackbar("cx", WINDOW, w // 2, w, lambda x: None)
    cv2.createTrackbar("cy", WINDOW, h // 2, h, lambda x: None)

    # Extrinsic translation (scaled by 100: [-5, 5] m)
    for name in ("tx", "ty", "tz"):
        cv2.createTrackbar(name, WINDOW, 500, 1000, lambda x: None)

    # Extrinsic rotation in degrees [-180, 180]
    for name in ("roll", "pitch", "yaw"):
        cv2.createTrackbar(name, WINDOW, 180, 360, lambda x: None)


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

    # Center point cloud at origin
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
    pcd.paint_uniform_color([0.0, 1.0, 0.0])
    vis.add_geometry(pcd)

    # Reference axes
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    vis.add_geometry(coord_frame)

    # Frustum geometry
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(np.zeros((5, 3)))
    frustum.lines = o3d.utility.Vector2iVector([
        [0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]
    ])
    frustum.colors = o3d.utility.Vector3dVector([[1.0,0.0,0.0]]*8)
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


def prompt_values():
    """Ask the user for numeric values and update the sliders."""
    root = tk.Tk()
    root.withdraw()
    names = ["fx", "fy", "cx", "cy", "tx", "ty", "tz", "roll", "pitch", "yaw"]
    vals = {}
    for name in names:
        v = simpledialog.askstring("Calibration Input", f"{name} =")
        if v is None:
            root.destroy()
            return
        vals[name] = float(v)
    vals["roll"] = np.deg2rad(vals["roll"])
    vals["pitch"] = np.deg2rad(vals["pitch"])
    vals["yaw"] = np.deg2rad(vals["yaw"])

    cv2.setTrackbarPos("fx", WINDOW, int(vals["fx"]))
    cv2.setTrackbarPos("fy", WINDOW, int(vals["fy"]))
    cv2.setTrackbarPos("cx", WINDOW, int(vals["cx"]))
    cv2.setTrackbarPos("cy", WINDOW, int(vals["cy"]))
    cv2.setTrackbarPos("tx", WINDOW, int(vals["tx"] * 100 + 500))
    cv2.setTrackbarPos("ty", WINDOW, int(vals["ty"] * 100 + 500))
    cv2.setTrackbarPos("tz", WINDOW, int(vals["tz"] * 100 + 500))
    cv2.setTrackbarPos("roll", WINDOW, int(np.rad2deg(vals["roll"]) + 180))
    cv2.setTrackbarPos("pitch", WINDOW, int(np.rad2deg(vals["pitch"]) + 180))
    cv2.setTrackbarPos("yaw", WINDOW, int(np.rad2deg(vals["yaw"]) + 180))
    root.destroy()


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

    print("Press 'i' for manual input, 's' to save, 'q' to quit.")
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
        cv2.putText(disp, "i: input, s: save, q: quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(WINDOW, disp)
        _update_frustum(frustum, K, T, img.shape)
        if vis is not None:
            vis.update_geometry(frustum)
            vis.poll_events()
            vis.update_renderer()
        key = cv2.waitKey(10) & 0xFF
        if key == ord("i"):
            prompt_values()
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