# ROS2 ADAS Perception and Tracking Stack

A **modular ROS2-based ADAS perception stack** built on the **nuScenes dataset**, implementing **LiDAR object detection, multi-object tracking, ego-relative velocity estimation, and Time-To-Collision (TTC) reasoning**.

The stack is designed with a **detector-agnostic architecture**, enabling seamless replacement of the perception frontend with:

* Ground-truth detections
* Classical LiDAR clustering
* Learned 3D detectors (**PointPillars**, **CenterPoint**)
* Future multi-sensor fusion modules

This project is structured as a **mini autonomy perception pipeline**, bridging **raw LiDAR sensing → object tracking → collision risk reasoning → downstream planning hooks**.

---

## Demo

![ADAS Demo](demo1.gif)

---

## 🚗 System Pipeline

```text
nuScenes → Detection → Data Association → Multi-Object Tracking
         → Velocity Estimation → Time-To-Collision → Collision Risk
```

Each stage is implemented as an independent **ROS2 node**, making the system highly modular and extensible.

---

## ✨ Core Features

* **ROS2 modular perception pipeline**
* **nuScenes dataset playback node**
* **LiDAR point cloud processing**
* **Ground-truth and DBSCAN-based object detection**
* **Kalman Filter-based multi-object tracking**
* **Hungarian algorithm for data association**
* **Per-object velocity estimation**
* **Ego-relative kinematic reasoning**
* **Time-To-Collision (TTC) computation**
* **RViz visualization of tracks, velocities, and TTC**
* **Plug-and-play detector backend design**
* Planned integration for **PointPillars** and **CenterPoint**

---

## 🧠 System Architecture

### Active ROS2 Nodes

### `nuscenes_player`

Publishes:

* LiDAR point clouds
* camera images
* ground-truth 3D detections
* ego pose and TF tree

### `lidar_cluster_detector`

Performs:

* point cloud preprocessing
* DBSCAN clustering
* centroid extraction
* bounding box estimation

### `gt_tracker_node`

Core perception intelligence node responsible for:

* track initialization
* Kalman Filter state updates
* Hungarian data association
* track velocity estimation
* TTC computation
* collision risk scoring hooks

### `lidar_detection_visualizer`

Publishes RViz markers for:

* 3D tracked boxes
* velocity arrows
* object IDs
* TTC labels
* collision warnings

---

## 🧩 Detector Abstraction Layer

The perception frontend follows a **detector-agnostic interface**, enabling rapid swapping between classical and learned detectors.

### Implemented

* Ground Truth Detector
* LiDAR Cluster Detector

### Planned

* PointPillars 3D Detector
* CenterPoint 3D Detector
* Camera-LiDAR Fusion Detector

This makes the stack highly suitable for **ADAS perception benchmarking**.

---

## 📂 Repository Structure

```text
ros2-adas-perception-stack/
│
├── src/
│   └── av_fusion/
│       ├── nuscenes_player.py
│       ├── lidar_cluster_detector.py
│       ├── gt_tracker_node.py
│       ├── lidar_detection_visualizer.py
│       ├── pointpillars_detector_node.py
│       └── centerpoint_detector_node.py
│
├── launch/
├── configs/
├── package.xml
├── setup.py
├── requirements.txt
└── README.md
```

---

## 📦 Dataset Setup

This project uses the **nuScenes dataset**.

Recommended for development:

* **nuScenes-mini**

Download from:
https://www.nuscenes.org/download

Expected dataset location:

```bash
~/av_perception/data/nuscenes
```

---

## ⚙️ Installation

### Clone repository

```bash
git clone https://github.com/GauravR2012/ros2-adas-perception-stack.git
cd ros2-adas-perception-stack
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Build workspace

```bash
colcon build
```

### Source workspace

```bash
source install/setup.bash
```

---

## ▶️ Running the Full Pipeline

### 1) Start nuScenes playback

```bash
ros2 run av_fusion nuscenes_player
```

### 2) Start detector (optional clustering mode)

```bash
ros2 run av_fusion lidar_cluster_detector
```

### 3) Start tracker + TTC node

```bash
ros2 run av_fusion gt_tracker
```

### 4) Start RViz visualizer

```bash
ros2 run av_fusion lidar_detection_visualizer
```

### 5) Launch RViz

```bash
rviz2
```

---

## 📊 Visualization Outputs

The RViz pipeline visualizes:

* LiDAR point clouds
* detected objects
* tracked trajectories
* unique track IDs
* velocity vectors
* TTC text overlays
* collision risk indicators

This enables **end-to-end perception debugging and ADAS scenario analysis**.

---

## 📈 Engineering Highlights

This repository demonstrates:

* **ROS2 middleware engineering**
* **real-time perception pipeline design**
* **tracking-by-detection systems**
* **kinematic state estimation**
* **ADAS collision reasoning**
* **modular detector abstraction**
* **downstream autonomy stack integration readiness**

---

## 🔮 Roadmap / Future Work

* Camera + LiDAR sensor fusion
* Learned 3D detector integration
* trajectory prediction
* lane-aware risk reasoning
* collision risk classification
* planner and control integration
* CARLA closed-loop simulation
* autonomous emergency braking prototype

---

## 🛠️ Technologies

* ROS2
* Python
* nuScenes
* NumPy
* SciPy
* OpenCV
* scikit-learn
* RViz
* Kalman Filtering
* Hungarian Matching
* DBSCAN

---

## 👨‍💻 Author

**Gaurav Ramteke**
Robotics | ADAS Perception | Autonomous Systems
