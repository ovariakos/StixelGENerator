from .BaseClasses import BaseData, CameraInfo, Pose
try:
    from .KittiDataset import KittiDataLoader, KittiData
    KITTI_IMPORT_ERROR = None
except ModuleNotFoundError as err:
    KittiDataLoader = None  # type: ignore
    KittiData = None  # type: ignore
    KITTI_IMPORT_ERROR = err
try:
    from .CoopScenes import CoopSceneData, CoopScenesDataLoader
    COOPSCENES_IMPORT_ERROR = None
except ModuleNotFoundError as err:
    CoopScenesDataLoader = None  # type: ignore
    CoopSceneData = None  # type: ignore
    COOPSCENES_IMPORT_ERROR = err
# from .CityscapesDataset import CityscapesDataLoader, CityscapesData
from .RosbagDataset import RosbagDataLoader, RosbagData

try:
    from .WaymoDataset import WaymoDataLoader, WaymoData
    WAYMO_IMPORT_ERROR = None
except ModuleNotFoundError as err:
    WaymoDataLoader = None  # type: ignore
    WaymoData = None  # type: ignore
    WAYMO_IMPORT_ERROR = err
