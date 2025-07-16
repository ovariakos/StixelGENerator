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
   List the available topics stored in the `.db3` database:
   ```bash
   sqlite3 output_bag.db3 "SELECT name FROM topics;"
   ```

3. **Extract messages**
   Use the `utility/rosbag_to_dataset.py` script to export files from the chosen image and point cloud topics. For example with a ZED2i left camera and an Ouster LiDAR:
   ```bash
   python utility/rosbag_to_dataset.py \
       --db output_bag.db3 \
       --image_topic /zed2i/zed_node/left/image_rect_color/compressed \
       --pc_topic /ouster/points \
       --out dataset_raw
   ```
   The utility numbers the extracted messages consecutively. Each frame consists
   of `images/<index>.jpg` and `pointclouds/<index>.csv`. A `dataset_map.csv`
   file stores the original ROS timestamps so that the order can easily be
   reconstructed later.

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
   Camera–LiDAR calibration (K, P and extrinsic) must be obtained from the
   rosbag or your sensor setup and referenced by your dataloader. The mapping
   provided by `dataset_map.csv` can be used to associate the calibration with
   each frame.

6. **Generate Stixel Worlds**
   Configure the new data path in `config.yaml` and start the generation:
   ```bash
   python generate.py
   ```

7. **Implement a dataloader**
   The repository does not ship a generic loader for rosbag exports. Use the
   provided `dataset_map.csv` as a starting point to write your own loader that
   reads the images, loads the CSV point clouds and applies your calibration
   matrices. Existing dataloaders in the `dataloader` directory can serve as
   examples.
