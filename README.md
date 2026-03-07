# ROS2 ADAS Perception and Tracking Stack

A modular **ROS2-based ADAS perception pipeline** built using the **nuScenes dataset**, implementing LiDAR-based object detection, multi-object tracking, velocity estimation, and Time-To-Collision (TTC) computation.

The project is designed as a **modular perception stack**, allowing easy replacement of detectors (ground truth, clustering, or learned models like PointPillars and CenterPoint).

---

## System Overview

Pipeline:

nuScenes → Detector → Tracker → Velocity Estimation → TTC → Collision Risk

Modules implemented as ROS2 nodes.

---

## Features

• ROS2-based modular perception pipeline  
• LiDAR point cloud processing  
• Ground-truth and clustering-based object detection  
• Multi-object tracking using Kalman Filter  
• Hungarian algorithm for data association  
• Ego-relative velocity estimation  
• Time-To-Collision (TTC) computation  
• RViz visualization of tracked objects, velocities, and TTC  
• Detector modularity for PointPillars and CenterPoint integration  

---

## Architecture

Nodes in the system:

nuscenes_player
→ publishes camera images, LiDAR, GT detections, TF

lidar_cluster_detector
→ DBSCAN clustering on LiDAR point clouds

gt_tracker_node
→ multi-object tracking + velocity + TTC

lidar_detection_visualizer
→ RViz visualization


Detector modules (modular interface):


Ground Truth Detector
LiDAR Cluster Detector
PointPillars Detector (planned)
CenterPoint Detector (planned)


---

## Repository Structure


ros2-adas-perception-stack
│
├── src/av_fusion
│ ├── gt_tracker_node.py
│ ├── nuscenes_player.py
│ ├── lidar_cluster_detector.py
│ ├── lidar_detection_visualizer.py
│ ├── pointpillars_detector_node.py
│ └── centerpoint_detector_node.py
│
├── package.xml
├── setup.py
├── README.md
└── requirements.txt


---

## Dataset

This project uses the **nuScenes dataset**.

Download:

https://www.nuscenes.org/download

Recommended version for testing:


nuScenes-mini


Place the dataset in:


~/av_perception/data/nuscenes


---

## Installation

### 1. Clone repository


git clone https://github.com/YOUR\_USERNAME/ros2-adas-perception-stack.git


### 2. Install dependencies


pip install -r requirements.txt


### 3. Build ROS2 workspace


colcon build


### 4. Source workspace


source install/setup.bash


---

## Running the Pipeline

### Start nuScenes player


ros2 run av_fusion nuscenes_player


### Start tracker


ros2 run av_fusion gt_tracker


### Start visualization


ros2 run av_fusion lidar_detection_visualizer


### Open RViz


rviz2


---

## Visualization

The system visualizes:

• LiDAR point clouds  
• Ground truth objects (green boxes)  
• Tracked objects (red boxes)  
• Velocity vectors (blue arrows)  
• Time-To-Collision labels  

---

## Future Work

Planned improvements:

• Integration of **PointPillars LiDAR detector**  
• Integration of **CenterPoint detector**  
• Sensor fusion (camera + LiDAR)  
• Collision risk classification  
• Motion prediction  
• Planning and control integration  

---

## Technologies Used

• ROS2  
• Python  
• nuScenes dataset  
• NumPy  
• SciPy  
• OpenCV  
• scikit-learn  
• RViz  

---

## Author

Adarsh Sharma

Robotics / Autonomous Systems Enthusiast

requirements.txt
numpy
scipy
opencv-python
scikit-learn
nuscenes-devkit
matplotlib
tqdm
pyyaml
Optional (recommended)

Add a .gitignore file too:

build/
install/
log/
.vscode/
__pycache__/
*.pyc
*.pyo
*.db
One improvement I strongly recommend

Add one RViz screenshot showing:

LiDAR point cloud

red tracked boxes

blue velocity arrows

TTC text

That instantly makes your repo look 5× more professional.

If you'd like, I can also give you a very polished GitHub repo structure used by robotics companies, which will make this project look significantly more industry-level.

Can you give me the readme.me in copy paste ready in one single cell

Here is a single copy-paste ready README.md. Just paste this directly into your GitHub README.md file.

# ROS2 ADAS Perception and Tracking Stack

A modular **ROS2-based ADAS perception pipeline** built using the **nuScenes dataset**, implementing LiDAR-based object detection, multi-object tracking, velocity estimation, and Time-To-Collision (TTC) computation.

The system is designed as a **modular perception stack**, allowing easy replacement of detectors (ground truth, clustering, or learned models such as PointPillars and CenterPoint).

---

# System Pipeline

nuScenes → Detector → Tracker → Velocity Estimation → TTC → Collision Risk

---

# Features

• ROS2-based modular perception pipeline  
• LiDAR point cloud processing  
• Ground truth and clustering-based object detection  
• Multi-object tracking using Kalman Filter  
• Hungarian algorithm for data association  
• Ego-relative velocity estimation  
• Time-To-Collision (TTC) computation  
• RViz visualization of tracked objects, velocities, and TTC  
• Modular detector interface for PointPillars and CenterPoint  

---

# ROS2 Nodes

### nuscenes_player
Publishes:
- Camera images
- LiDAR point clouds
- Ground truth detections
- TF transforms

### lidar_cluster_detector
Performs DBSCAN clustering on LiDAR point clouds to generate detections.

### gt_tracker_node
Performs:
- Multi-object tracking
- Velocity estimation
- TTC computation

### lidar_detection_visualizer
Visualizes detections and tracking results in RViz.

---

# Repository Structure


ros2-adas-perception-stack
│
├── src
│ └── av_fusion
│ ├── gt_tracker_node.py
│ ├── nuscenes_player.py
│ ├── lidar_cluster_detector.py
│ ├── lidar_detection_visualizer.py
│ ├── pointpillars_detector_node.py
│ └── centerpoint_detector_node.py
│
├── package.xml
├── setup.py
├── README.md
└── requirements.txt


---

# Dataset

This project uses the **nuScenes dataset**.

Download from:  
https://www.nuscenes.org/download

Recommended dataset for testing:


nuScenes-mini


Place dataset in:


~/av_perception/data/nuscenes


---

# Installation

Clone the repository:


git clone https://github.com/YOUR_USERNAME/ros2-adas-perception-stack.git


Install dependencies:


pip install -r requirements.txt


Build ROS2 workspace:


colcon build


Source workspace:


source install/setup.bash


---

# Running the System

Start nuScenes player:


ros2 run av_fusion nuscenes_player


Start tracker:


ros2 run av_fusion gt_tracker


Start visualization node:


ros2 run av_fusion lidar_detection_visualizer


Open RViz:


rviz2


---

# Visualization

The system visualizes:

• LiDAR point cloud  
• Ground truth objects (green boxes)  
• Tracked objects (red boxes)  
• Velocity vectors (blue arrows)  
• Time-To-Collision (TTC) labels  

---

# Future Work

• Sensor fusion (camera + LiDAR)  
• Collision risk classification  
• Motion prediction  
• Planning and control integration  

---

# Technologies Used

ROS2  
Python  
nuScenes dataset  
NumPy  
SciPy  
OpenCV  
scikit-learn  
RViz  

---


