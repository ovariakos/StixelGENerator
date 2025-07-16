# Stixel World Generator from LiDAR data
This repo is designed to generate custom Stixel training data based on different datasets. And is related to:

[Toward Monocular Low-Weight Perception for Object Segmentation and Free Space Detection](https://ieeexplore.ieee.org/Xplore/home.jsp). IV 2024.\
[Marcel Vosshans](https://scholar.google.de/citations?user=_dbcdr4AAAAJ&hl=en), [Omar Ait-Aider](https://scholar.google.fr/citations?user=NIdLQnUAAAAJ&hl=en), [Youcef Mezouar](https://youcef-mezouar.wixsite.com/ymezouar) and [Markus Enzweiler](https://markus-enzweiler.de/)\
University of Esslingen, UCA Sigma Clermont\
[[`Xplore`](https://ieeexplore.ieee.org/Xplore/home.jsp)]
## StixelGENerator
![Sample Stixel World by LiDAR](/docs/imgs/sample_stixel_world.png)
This repo provides the basic toolset to generate a Stixel World from LiDAR. It is used as Ground Truth for 
the [StixelNExT](https://github.com/MarcelVSHNS/StixelNExT) 2D estimator as well as for the 3D approach: [StixelNExT++](https://github.com/MarcelVSHNS/StixelNExT_Pro).

### Usage with Waymo or KITTI
1. Clone the repo to your local machine
2. Set up a virtual environment with `python -m venv venv` (we tested on Python 3.10) or Anaconda respectively `conda create -n StixelGEN python=3.10`. Activate with `source venv/bin/activate`/ `conda activate StixelGEN`
3. Install requirements with `pip install -r requirements.txt`/ `conda install --file requirements.txt` 
4. Configure the project: adapt your paths in the `config.yaml`-file and select the dataset with the import in `/generate.py` like:
```python
from dataloader import WaymoDataLoader as Dataset   # or KittiDataLoader
```
After that you can test the functionalities with `utility/explore.py` or run `/generate.py` to generate Stixel Worlds.

#### Output
The output is a `.stx1` file which is a Protobuf for Stixel Worlds. It includes the Stixel as well as the image and is 
ready to serve as input data for StixelNExT++. The corresponding library can be installed with `pip install pyStixel-lib`
and is public available [here](https://github.com/MarcelVSHNS/pyStixel-lib).

#### KITTI Training Data
We also provide an already generated dataset, based on the public available KITTI dataset. It can be downloaded
[here](https://drive.google.com/drive/folders/1ft99z9F4053zDzyIDn2DZ_8qh5if-QvW?usp=sharing) (35.48 GB)

### Adaption to other Datasets
The repo is designed to work with adaptive dataloader, which can be handled by the import commands. 
For your own usage its necessary to write a new Dataloader for what the `dataloader/BaseLoader` and 
`dataloader/BaseData` can be used. Needed synchronised data/information are:
* The camera image
* LiDAR data
* Camera calibration data: 
  * Camera Matrix
  * Projection Matrix
  * Rectify Matrix
  * Transformation Matrix
* Context information (image size, dataset name, ...)
* OPTIONAL: Stereo Images (right camera)
* OPTIONAL: LiDAR Calibration (in case of global T Matrices)

#### Fine tuning
You can heavily increase the results with the parameters from `libraries/pcl-config.yaml`. 
Documentation for the functions are provided by the code. The los (line of sight) parameter can cause huge holes!

### Utilities
* explore: A simple workspace script to use and inspect the derived data, step by step.

### Creating a dataset from rosbag
The following steps show how to extract a raw dataset from a rosbag so that it can be processed by StixelGENerator.

1. **Convert the rosbag**
   If your recording is in ROS1 format, convert it to ROS2 first:
   ```bash
   ros2 bag convert input.bag -o output_bag
   ```

2. **Inspect the bag contents**
   For SQLite-based ROS2 bags (`.db3`), list the available topics with:
   ```bash
   sqlite3 output_bag.db3 "SELECT name FROM topics;"
   ```
   For MCAP files, you can inspect channels using the
   [`mcap` command line tool](https://github.com/foxglove/mcap) or another viewer.

3. **Extract messages**
   Use the `utility/rosbag_to_dataset.py` script to export files from the chosen image and point cloud topics. The `--bag` argument accepts either a `.db3` ROS2 bag or an `.mcap` file. For example with a ZED2i left camera and an Ouster LiDAR:
   ```bash
  python utility/rosbag_to_dataset.py \
      --bag output_bag.db3 \
      --image_topic /zed2i/zed_node/left/image_rect_color/compressed \
      --pc_topic /ouster/points \
      --out dataset_raw
   ```
   The utility numbers the extracted messages consecutively and keeps only
   frames that contain both an image and a point cloud. Each pair is listed in
   `dataset_map.csv` together with the original ROS timestamps so that the
   capture order can be reconstructed later.

   The resulting folder tree therefore looks like:

   ```text
   dataset_raw/
       dataset_map.csv
       images/
           000000.jpg
           000001.jpg
           ...
       pointclouds/
           000000.csv
           000001.csv
           ...
   ```

4. **Verify compatibility**
   StixelGENerator's dataloaders can read JPEG images and `x,y,z` CSV files when
   calibration matrices are provided. Use the `dataset_map.csv` file to confirm
   that every index has both an image and a point cloud. Check that the
   coordinates are reasonable for your sensor setup.

5. **Add calibrations**
   Place a file named `calibration.yaml` in the same directory as
   `dataset_map.csv` (e.g. `dataset_raw/`). This file describes the
   camera–LiDAR calibration used by the dataloader:

   - `K` (3×3): camera matrix
   - `P` (3×4): projection matrix
   - `R` (4×4): rectification matrix, usually the identity
   - `T` (4×4): extrinsic transform from the camera to the LiDAR frame

   Example:

   ```yaml
   K: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
   P: [[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]]
   R: [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
   T: [[r11,r12,r13,tx], [r21,r22,r23,ty], [r31,r32,r33,tz], [0,0,0,1]]
   ```

   If `calibration.yaml` is missing, the loader falls back to identity
   matrices. The mapping provided by `dataset_map.csv` can then be used to
   associate the calibration with each frame.

6. **Tweak calibration**
   To manually align the point cloud with the image you can use
   `utility/calibration_gui.py`:
   ```bash
   python utility/calibration_gui.py --data dataset_raw
   ```
   Sliders allow adjusting the intrinsic and extrinsic parameters. Press `s` to
   write the current values to `calibration.yaml` in the dataset folder.

7. **Generate Stixel Worlds**
   Configure the new data path in `config.yaml` and start the generation:
   ```bash
   python generate.py
   ```

8. **Implement a dataloader**
   A simple loader named `RosbagDataLoader` is available to read this structure.
   It expects a `calibration.yaml` next to `dataset_map.csv` containing the
   camera matrices (K, P, R, T). You can use it directly by setting
   `dataset: rosbag` in `config.yaml` or adapt it for your own needs. Existing
   dataloaders in the `dataloader` directory can serve as examples.
