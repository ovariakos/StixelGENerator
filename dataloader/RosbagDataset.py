import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import List

from dataloader import BaseData, CameraInfo
from libraries.Stixel import point_dtype


class RosbagData(BaseData):

    def __init__(self, name: str, img_path: str, pc_path: str, cam_info: CameraInfo):
        super().__init__()
        self.name = name
        self.image = Image.open(img_path)
        self.camera_info = cam_info
        self.camera_pose = cam_info.T[:3, 3].reshape(3, 1)

        pts = np.loadtxt(pc_path, delimiter=",", skiprows=1)
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])

        proj = cam_info.P.dot(cam_info.R.dot(cam_info.T.dot(pts_h.T)))
        with np.errstate(divide="ignore", invalid="ignore"):
            proj[:2] = np.divide(
                proj[:2],
                proj[2, :],
                out=np.full_like(proj[:2], np.nan),
                where=proj[2, :] != 0,
            )

        proj = proj.T
        valid_mask = np.isfinite(proj).all(axis=1)
        proj = proj[valid_mask]
        pts = pts[valid_mask]
        # Clip projected points into signed 32-bit integer range
        i32_min, i32_max = -(2 ** 31), 2 ** 31 - 1
        proj = np.clip(proj, i32_min, i32_max).astype(np.int32)
        sem_seg = np.zeros((pts.shape[0], 1))
        combined = np.hstack([pts, proj, sem_seg])
        self.points = np.array([tuple(row) for row in combined], dtype=point_dtype)


class RosbagDataLoader:
    """DataLoader for rosbag exports produced by ``rosbag_to_dataset.py``."""

    def __init__(self, data_dir: str, first_only: bool = False):
        self.name = "rosbag"
        self.first_only = first_only
        self.data_dir = data_dir
        map_path = os.path.join(data_dir, "dataset_map.csv")
        self.record_map = pd.read_csv(map_path)
        with open(f"dataloader/configs/{self.name}-pcl-config.yaml") as yaml_file:
            self.config = yaml.safe_load(yaml_file)

        calib_path = os.path.join(data_dir, "calibration.yaml")
        if os.path.exists(calib_path):
            with open(calib_path) as f:
                calib_cfg = yaml.safe_load(f)
            K = np.array(calib_cfg.get("K", np.eye(3)))
            P = np.array(calib_cfg.get("P", np.hstack((K, np.zeros((3, 1))))) )
            T = np.array(calib_cfg.get("T", np.eye(4)))
            R = np.array(calib_cfg.get("R", np.eye(4)))
        else:
            K = np.eye(3)
            P = np.hstack((K, np.zeros((3, 1))))
            T = np.eye(4)
            R = np.eye(4)

        self.calib = CameraInfo(camera_mtx=K, trans_mtx=T, proj_mtx=P, rect_mtx=R)

        first_img = Image.open(os.path.join(data_dir, self.record_map.loc[0, "image_file"]))
        self.img_size = {"width": first_img.width, "height": first_img.height}
        print("DEBUG ÜZENET!!! 3")

    def __len__(self) -> int:
        return len(self.record_map)

    def __getitem__(self, idx: int) -> List[RosbagData]:
        print("DEBUG ÜZENET!!! 4")
        row = self.record_map.iloc[idx]
        print("DEBUG ÜZENET!!! 5")
        name = f"frame_{int(row['index']):06d}"
        print("DEBUG ÜZENET!!! 6")
        img_path = os.path.join(self.data_dir, row["image_file"])
        print("DEBUG ÜZENET!!! 7")
        pc_path = os.path.join(self.data_dir, row["pc_file"])
        print("DEBUG ÜZENET!!! 8")
        data = RosbagData(name, img_path, pc_path, self.calib)
        print("DEBUG ÜZENET!!! 9")
        return [data]
