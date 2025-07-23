#!/usr/bin/env python3
"""Interactive calibration tool for rosbag datasets.

This script overlays LiDAR points onto a camera image and allows the user to
adjust the camera intrinsics and the extrinsic transform via OpenCV sliders
or GUI dialogs. The resulting calibration will be written to ``calibration.yaml``
in the dataset directory when pressing ``s``.

Example:
    python calibration_gui.py --data /path/to/dataset --index 0

The dataset directory must contain ``dataset_map.csv`` and ``images``/``pointclouds``
subdirectories as created by ``rosbag_to_dataset.py``.
"""

import os
import argparse
import yaml
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import simpledialog

WINDOW = "calibration"


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

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
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


def prompt_values():
    """GUI párbeszédablakból értékbevitel és trackbar-frissítés."""
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
    # fokból radián
    vals["roll"]  = np.deg2rad(vals["roll"])
    vals["pitch"] = np.deg2rad(vals["pitch"])
    vals["yaw"]   = np.deg2rad(vals["yaw"])

    # Trackbar frissítése
    cv2.setTrackbarPos("fx",   WINDOW, int(vals["fx"]))
    cv2.setTrackbarPos("fy",   WINDOW, int(vals["fy"]))
    cv2.setTrackbarPos("cx",   WINDOW, int(vals["cx"]))
    cv2.setTrackbarPos("cy",   WINDOW, int(vals["cy"]))
    cv2.setTrackbarPos("tx",   WINDOW, int(vals["tx"]*100 + 500))
    cv2.setTrackbarPos("ty",   WINDOW, int(vals["ty"]*100 + 500))
    cv2.setTrackbarPos("tz",   WINDOW, int(vals["tz"]*100 + 500))
    cv2.setTrackbarPos("roll", WINDOW, int(np.rad2deg(vals["roll"]) + 180))
    cv2.setTrackbarPos("pitch",WINDOW, int(np.rad2deg(vals["pitch"]) + 180))
    cv2.setTrackbarPos("yaw",  WINDOW, int(np.rad2deg(vals["yaw"]) + 180))
    root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Manual calibration tool")
    parser.add_argument("--data", required=True, help="Dataset directory")
    parser.add_argument("--index", type=int, default=0, help="Frame index")
    args = parser.parse_args()

    img, pts = load_frame(args.data, args.index)
    create_trackbars(img.shape)

    print("Nyomj 'i'-t az értékbevitelhez, 's'-t a mentéshez, 'q'-t a kilépéshez.")
    while True:
        K, P, T = get_values(img.shape)
        uv, depth = project_points(pts, K, T)
        disp = overlay_points(img, uv, depth)
        # útmutató szöveg az ablakon
        cv2.putText(disp, "i: input, s: save, q: quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow(WINDOW, disp)
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

if __name__ == "__main__":
    main()