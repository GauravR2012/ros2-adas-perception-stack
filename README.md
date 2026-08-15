# ROS 2 ADAS Perception & Autonomous Driving Research Stack

A modular, research-oriented **ROS 2 ADAS perception stack** built around the **nuScenes dataset**, combining LiDAR perception, camera perception, semantic segmentation, lane understanding, optical flow, multi-object tracking, ego-motion reasoning, and collision-risk estimation.

The project is designed as an experimental **autonomous driving perception stack** rather than a single monolithic detector. Each perception capability is implemented as an independent ROS 2 node with explicit topic interfaces, allowing individual algorithms to be replaced, benchmarked, and extended.

The long-term objective is to evolve the stack from raw sensors to downstream planning/control:

```
Raw Sensors
  │
  ├── Camera
  └── LiDAR
       │
       ▼
   Perception
       │
       ├── Semantic Segmentation
       ├── Road Segmentation
       ├── Lane Detection
       ├── Optical Flow
       └── 3D Object Detection
       │
       ▼
Tracking & State Estimation
       │
       ├── Data Association
       ├── Multi-Object Tracking
       ├── Velocity Estimation
       └── Ego-relative State
       │
       ▼
Scene Understanding
       │
       ├── Lane Geometry
       ├── TTC
       └── Collision Risk
       │
       ▼
Downstream Planning / Control
```

---

## Project Status

> **Active research and implementation project**

The stack currently contains both **classical computer-vision methods** and **learned perception models**, allowing direct comparison between traditional robotics approaches and modern deep-learning approaches.

### Implemented

- ROS 2 modular architecture
- nuScenes Mini playback
- Camera playback
- LiDAR playback
- Ground-truth 3D bounding boxes
- RViz ground-truth visualization
- Semantic segmentation
- Binary road segmentation
- Classical lane detection
- Temporal lane tracking
- Lane-pair geometry consistency
- Lane-center offset estimation
- Lane heading estimation
- Sparse optical flow
- Shi-Tomasi feature detection
- Pyramidal Lucas-Kanade tracking
- RAFT-Small dense optical flow
- CPU-only RAFT inference
- LiDAR clustering
- Kalman-filter-based tracking infrastructure
- Hungarian data association
- Velocity estimation
- Time-To-Collision reasoning
- RViz visualization

### In Development

- Quantitative optical-flow evaluation
- Lane-aware object reasoning
- Camera-LiDAR association
- Learned 3D object detection
- Temporal fusion
- Improved ego-motion estimation
- Sensor fusion

### Planned

- PointPillars
- CenterPoint
- Camera-LiDAR fusion
- Trajectory prediction
- Lane-aware collision prediction
- Planning interfaces
- Closed-loop simulation
- Autonomous Emergency Braking prototype

---

## System Architecture

```
nuScenes Mini
      │
      ▼
┌───────────────────┐
│  nuScenes Player   │
└─────────┬──────────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
  Camera       LiDAR
    │             │
    ▼             ▼
 Semantic     LiDAR Cluster
Segmentation    Detection
    │             │
    ▼             ▼
Road Mask     3D Objects
    │             │
    ▼             ▼
Lane Detection  Tracking
    │             │
    ▼             ▼
Lane Geometry   Velocity
    │             │
    └──────┬──────┘
           ▼
   Scene Understanding
           │
    ┌──────┴──────┐
    ▼             ▼
Optical Flow     TTC
    │             │
    └──────┬──────┘
           ▼
     Risk Reasoning
           │
           ▼
   Planning / Control
```

---

## 1. Dataset Playback

### `nuscenes_player`

The dataset playback node provides a ROS 2 interface to the nuScenes dataset.

**Publishes**

- `/camera/front/image`
- `/lidar/points`
- `/detections/boxes_3d`
- `/gt/visualization_markers`
- `/tf`

**Responsibilities**

- nuScenes scene playback
- camera image publishing
- LiDAR point-cloud publishing
- ground-truth 3D annotation publishing
- ego-pose / TF publication
- RViz visualization support
- scene looping for repeated experiments

The player is intentionally kept as a dataset-facing sensor source so that downstream perception nodes remain independent of the dataset API.

The current development workflow uses a short nuScenes scene and loops it continuously so that perception modules can be repeatedly tested without requiring a long sequence.

---

## 2. Camera Semantic Segmentation

### `semantic_segmentation_node`

The semantic segmentation stage performs pixel-wise scene understanding from the front camera.

**Input**
- `/camera/front/image`

**Outputs**
- `/camera/segmentation/mask`
- `/camera/segmentation/overlay`

The segmentation output provides the semantic foundation for downstream road and lane reasoning.

---

## 3. Road Segmentation

### `road_segmentation_node`

The road segmentation node converts the semantic segmentation output into a binary road mask.

```
Semantic Segmentation
        │
        ▼
  Class ID Filtering
        │
        ▼
   Binary Road Mask
```

**Input**
- `/camera/segmentation/mask`

**Outputs**
- `/camera/segmentation/road_mask`
- `/camera/segmentation/road_overlay`

The road mask is represented as:

- `255` → road
- `0` → non-road

This creates a spatial prior that constrains the subsequent lane detector.

The road segmentation stage was introduced specifically so that lane detection does not have to operate over the entire camera image. Instead, lane extraction is restricted to the estimated road region.

---

## 4. Lane Detection

### `lane_detection_node`

The lane detector combines classical computer vision with the learned road segmentation output.

**Inputs**
- `/camera/front/image`
- `/camera/segmentation/road_mask`

**Outputs**
- `/camera/lane/overlay`
- `/camera/lane/mask`

**Processing Pipeline**

```
Camera Image
      │
      ▼
   Road Mask
      │
      ▼
   Road ROI
      │
      ▼
Lane-Marking Extraction
      │
      ├── White markings
      └── Yellow markings
      │
      ▼
Morphological Filtering
      │
      ▼
Canny Edge Detection
      │
      ▼
Hough Line Transform
      │
      ▼
Left / Right Lane Classification
      │
      ▼
   Lane Fitting
      │
      ▼
Temporal Tracking
      │
      ▼
Lane-Pair Geometry
```

The road segmentation mask is used as a spatial constraint before lane extraction.

This substantially reduces irrelevant image regions and allows the classical lane detector to focus on the drivable road surface.

---

## 5. Temporal Lane Tracking

The lane detector does not rely entirely on frame-by-frame detection.

Temporal filtering is used to stabilize the lane estimates when individual frames contain weak or missing lane detections.

Current mechanisms include:

- Exponential Moving Average
- missed-frame handling
- temporal consistency checks
- bottom-x jump constraints
- slope consistency
- lane-pair validation
- temporal geometry smoothing

Conceptually:

```
Current Frame
      │
      ▼
Raw Lane Detection
      │
      ▼
Temporal Consistency Check
      │
      ├── Valid ────────► Update Track
      │
      └── Invalid ──────► Reject / Retain Previous Estimate
      │
      ▼
Temporal Estimate
```

This allows the system to maintain a lane estimate when one side is temporarily missed.

The temporal tracker was introduced before moving toward optical-flow-based temporal reasoning, providing a classical baseline for temporal lane stability.

---

## 6. Lane Geometry

The lane detector also estimates higher-level geometric quantities.

**Current geometric outputs include**

- left lane
- right lane
- lane width
- lane-center position
- lateral offset
- image-space heading
- lane-width consistency
- rejected candidate statistics

Example diagnostic output:

```
Lane geometry |
width=1025px |
offset=-149px |
heading=-174.6deg |
width_std=253.6px
```

These quantities provide an interface between low-level lane perception and higher-level driving reasoning.

The lane geometry stage is particularly important for future lane-aware TTC and collision-risk reasoning.

---

## 7. Sparse Optical Flow

### `optical_flow_node`

A classical sparse optical-flow baseline is implemented using:

- Shi-Tomasi Corner Detection
- Pyramidal Lucas-Kanade Optical Flow

**Pipeline**

```
Frame t
      │
      ▼
Shi-Tomasi Features
      │
      ▼
Frame t+1
      │
      ▼
Lucas-Kanade Tracking
      │
      ▼
Tracked Feature Correspondences
      │
      ▼
Motion Statistics
```

The implementation provides a lightweight classical baseline for temporal motion estimation.

Typical statistics include:

- number of tracked features
- mean flow magnitude
- median flow magnitude
- maximum flow magnitude

This provides a computationally inexpensive baseline before introducing learned dense optical flow.

---

## 8. Learned Dense Optical Flow

### `raft_optical_flow_node`

The project also includes **RAFT-Small** as a learned dense optical-flow backend.

**Model:** RAFT-Small

**Current Configuration**

- Device: CPU
- Input Resolution: 640 × 360

The model is loaded using the Torchvision optical-flow API.

**Pipeline**

```
Frame t                Frame t+1
   │                       │
   └───────────┬───────────┘
               ▼
          RAFT-Small
               │
               ▼
         Dense Flow Field
               │
        ┌──────┴──────┐
        ▼             ▼
    Magnitude      Direction
               │
               ▼
        Motion Statistics
```

The current CPU implementation runs at approximately:

- ~0.3–0.35 FPS
- ~2.8–3.0 seconds inference

at 640×360 on the development machine.

The node includes:

- dense optical-flow estimation
- robust flow statistics
- timestamp monitoring
- scene-loop handling
- latest-frame buffering
- stale-frame dropping

The latest-frame buffering strategy prevents RAFT from accumulating an ever-growing queue of stale camera frames.

Because the nuScenes player loops the scene, the RAFT node also detects scene-loop timestamp discontinuities and resets its temporal state when the sequence jumps backward.

---

## 9. Classical vs Learned Optical Flow

The project intentionally maintains both classical and learned optical-flow implementations.

```
   Classical                    Learned
Shi-Tomasi + Lucas-Kanade       RAFT-Small
        │                           │
        ▼                           ▼
   Sparse Flow                 Dense Flow
```

This allows future experimental comparison across:

- runtime
- feature density
- motion magnitude
- temporal consistency
- robustness
- computational cost
- CPU resource requirements

The objective is not simply to demonstrate a deep-learning model, but to understand the trade-offs between classical robotics algorithms and learned perception.

---

## 10. LiDAR Object Detection

### `lidar_cluster_detector`

A classical LiDAR perception backend based on spatial clustering is included.

**Processing**

```
LiDAR Point Cloud
      │
      ▼
Point Filtering
      │
      ▼
DBSCAN Clustering
      │
      ▼
Cluster Centroids
      │
      ▼
3D Bounding Box Estimation
```

This provides a lightweight object-detection baseline without requiring a learned 3D detector.

The detector is intentionally implemented as a replaceable frontend so that learned detectors can later be introduced without redesigning the downstream tracking stack.

---

## 11. Detector-Agnostic Architecture

The perception frontend is designed so that the downstream tracking layer does not depend on a specific detector implementation.

Conceptually:

```
                     Detector Interface
                            │
      ┌───────────────┬─────────────┬───────────────┐
      ▼                ▼             ▼               ▼
Ground Truth        DBSCAN        LiDAR        Learned 3D
                    LiDAR                        Detector
      │                │             │               │
      └────────────────┴─────────────┴───────────────┘
                            ▼
                    Object Detections
                            │
                            ▼
                         Tracker
```

Potential detector backends include:

- Ground Truth
- DBSCAN LiDAR Clustering
- PointPillars
- CenterPoint
- Camera-LiDAR Fusion

The downstream tracker can therefore be benchmarked independently of the detector.

---

## 12. Multi-Object Tracking

### `gt_tracker_node`

The tracking stage follows a tracking-by-detection architecture.

**Core Components**

- track initialization
- state prediction
- measurement update
- Kalman filtering
- Hungarian data association
- track management
- velocity estimation

**Pipeline**

```
Detections
      │
      ▼
Prediction
      │
      ▼
Cost Matrix
      │
      ▼
Hungarian Assignment
      │
      ▼
Kalman Update
      │
      ▼
Tracks
```

This separates instantaneous object detection from persistent object state estimation.

---

## 13. Velocity Estimation

Tracked object states are used to estimate object velocity.

The system supports reasoning about:

```
Object Velocity + Ego Velocity
              │
              ▼
      Relative Velocity
```

This provides the basis for downstream collision-risk estimation.

Future work will improve this into a more explicit ego-relative state-estimation layer with uncertainty propagation.

---

## 14. Time-To-Collision

The tracking layer provides a foundation for Time-To-Collision reasoning.

For a simplified longitudinal model:

```
TTC = -d / v_rel
```

where:

- `d` = relative longitudinal distance
- `v_rel` = relative longitudinal velocity

The collision-risk layer can subsequently be extended toward:

- lane-aware TTC
- 2D/3D collision geometry
- uncertainty-aware TTC
- object-specific risk thresholds
- probabilistic collision prediction

---

## 15. Ground-Truth Visualization

The nuScenes player publishes ground-truth 3D bounding boxes for evaluation and debugging.

The ground-truth markers are visualized in RViz using green 3D bounding boxes.

This provides a direct visual reference for comparing:

- Ground Truth
- Detection
- Tracking

against the same scene.

Ground-truth visualization is particularly useful when debugging:

- object detection
- coordinate frames
- TF alignment
- tracker behavior
- data association
- scene playback

---

## 16. RViz2 Visualization

RViz2 is used as the primary debugging and evaluation interface.

Current visualization includes:

- camera stream
- semantic segmentation
- road mask
- lane overlay
- optical-flow visualization
- LiDAR point clouds
- ground-truth boxes
- detected boxes
- tracked boxes
- object IDs
- velocity vectors
- TTC information

The visualization layer is deliberately separated from the perception algorithms so that individual outputs can be inspected independently.

---

## ROS 2 Topic Architecture

| Topic | Type | Purpose |
|---|---|---|
| `/camera/front/image` | `sensor_msgs/msg/Image` | Front camera |
| `/camera/segmentation/mask` | `sensor_msgs/msg/Image` | Semantic segmentation |
| `/camera/segmentation/road_mask` | `sensor_msgs/msg/Image` | Binary road mask |
| `/camera/segmentation/overlay` | `sensor_msgs/msg/Image` | Segmentation visualization |
| `/camera/segmentation/road_overlay` | `sensor_msgs/msg/Image` | Road-mask visualization |
| `/camera/lane/overlay` | `sensor_msgs/msg/Image` | Lane visualization |
| `/camera/lane/mask` | `sensor_msgs/msg/Image` | Lane mask |
| `/lidar/points` | `sensor_msgs/msg/PointCloud2` | LiDAR point cloud |
| `/detections/boxes_3d` | `vision_msgs/msg/Detection3DArray` | 3D detections |
| `/gt/visualization_markers` | `visualization_msgs/msg/MarkerArray` | Ground-truth visualization |
| `/tf` | `tf2_msgs/msg/TFMessage` | Transform tree |

---

## Repository Structure

```
ros2-adas-perception-stack/
│
├── src/
│   └── av_fusion/
│       ├── av_fusion/
│       │   ├── nuscenes_player.py
│       │   │
│       │   ├── semantic_segmentation_node.py
│       │   ├── road_segmentation_node.py
│       │   ├── lane_detection_node.py
│       │   │
│       │   ├── optical_flow_node.py
│       │   ├── raft_optical_flow_node.py
│       │   │
│       │   ├── lidar_cluster_detector.py
│       │   ├── gt_tracker_node.py
│       │   ├── lidar_detection_visualizer.py
│       │   │
│       │   ├── pointpillars_detector_node.py
│       │   └── centerpoint_detector_node.py
│       │
│       ├── package.xml
│       ├── setup.py
│       └── resource/
│
├── launch/
├── configs/
├── requirements.txt
├── README.md
└── demo1.gif
```

> The repository structure may evolve as new experimental modules are added.

---

## Dataset

This project uses the **nuScenes autonomous-driving dataset**.

For development and experimentation, the project uses:

- nuScenes Mini

Dataset: https://www.nuscenes.org/download

Expected local structure:

```
~/av_perception/data/nuscenes/
├── samples/
├── sweeps/
├── maps/
├── v1.0-mini/
└── ...
```

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/GauravR2012/ros2-adas-perception-stack.git
cd ros2-adas-perception-stack
```

### 2. Source ROS 2

The current development environment uses:

- ROS 2 Jazzy
- Linux
- Python 3

```bash
source /opt/ros/jazzy/setup.bash
```

### 3. Create Python Environment

The camera perception and deep-learning components use a dedicated Python environment.

```bash
cd ~/ros2_ws
python3 -m venv segmentation_env
source ~/ros2_ws/segmentation_env/bin/activate
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

For the RAFT-Small implementation, the environment requires PyTorch and Torchvision.

Example verification:

```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

The current development setup uses CPU inference.

### 5. Build ROS 2 Workspace

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build
```

Then:

```bash
source ~/ros2_ws/install/setup.bash
```

---

## Running the Stack

The system is intentionally modular. Individual nodes can be started independently for debugging and experimentation.

### Terminal 1 — nuScenes Player

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/segmentation_env/bin/activate

python ~/ros2_ws/src/av_fusion/av_fusion/nuscenes_player.py
```

The player loops the selected nuScenes scene so that the short development sequence can be repeatedly evaluated.

### Terminal 2 — Semantic Segmentation

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/segmentation_env/bin/activate

python ~/ros2_ws/src/av_fusion/av_fusion/semantic_segmentation_node.py
```

### Terminal 3 — Road Segmentation

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/segmentation_env/bin/activate

python ~/ros2_ws/src/av_fusion/av_fusion/road_segmentation_node.py
```

### Terminal 4 — Lane Detection

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/segmentation_env/bin/activate

python ~/ros2_ws/src/av_fusion/av_fusion/lane_detection_node.py
```

### Terminal 5 — Classical Optical Flow

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/segmentation_env/bin/activate

python ~/ros2_ws/src/av_fusion/av_fusion/optical_flow_node.py
```

### Terminal 6 — RAFT-Small Optical Flow

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/segmentation_env/bin/activate

python ~/ros2_ws/src/av_fusion/av_fusion/raft_optical_flow_node.py
```

### Terminal 7 — LiDAR Cluster Detection

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/segmentation_env/bin/activate

python ~/ros2_ws/src/av_fusion/av_fusion/lidar_cluster_detector.py
```

### Terminal 8 — Tracking

If the tracker is installed as a ROS 2 executable:

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 run av_fusion gt_tracker
```

### Terminal 9 — RViz2

```bash
source /opt/ros/jazzy/setup.bash
rviz2
```

---

## Recommended Experimental Startup Order

**Camera Perception**

1. nuScenes Player
2. Semantic Segmentation
3. Road Segmentation
4. Lane Detection

**Classical Optical Flow**

1. nuScenes Player
2. `optical_flow_node`

**RAFT Optical Flow**

1. nuScenes Player
2. `raft_optical_flow_node`

**LiDAR Tracking**

1. nuScenes Player
2. LiDAR Detector
3. Tracker
4. RViz

---

## Experimental Workflow

**Camera Perception**

```
nuScenes → Camera → Semantic Segmentation → Road Mask → Lane Detection → Temporal Lane Geometry
```

**Motion Perception**

```
Camera → Shi-Tomasi + Lucas-Kanade ─┐
      └→ RAFT-Small ────────────────┴→ Temporal Motion
```

**LiDAR Perception**

```
LiDAR → Clustering / 3D Detection → Tracking → Velocity → TTC
```

**Future Multimodal Stack**

```
Camera ──────────────┐
                      ▼
              Camera-LiDAR Fusion
                      ▲
LiDAR ────────────────┘
                      │
                      ▼
              Unified 3D Scene
                      │
                      ▼
                  Tracking
                      │
              ┌───────┴───────┐
              ▼               ▼
       Lane Geometry         TTC
              │               │
              └───────┬───────┘
                      ▼
               Risk Reasoning
                      │
                      ▼
              Planning / Control
```

---

## Evaluation Strategy

The project is intended to move beyond qualitative RViz demonstrations toward quantitative evaluation.

**Semantic Segmentation** — planned metrics:
- mean Intersection-over-Union (mIoU)
- pixel accuracy
- class-wise IoU

**Lane Detection** — planned metrics:
- lane detection precision
- lane detection recall
- lateral offset error
- heading error
- lane-width consistency
- temporal stability

**Optical Flow** — metrics:
- Endpoint Error (EPE)
- angular error
- flow outlier percentage
- runtime / FPS
- CPU utilization
- temporal consistency

**3D Detection** — metrics:
- mean Average Precision (mAP)
- translation error
- scale error
- orientation error
- velocity error

**Multi-Object Tracking** — metrics:
- MOTA
- MOTP
- ID switches
- track fragmentation
- velocity RMSE

**Collision Risk** — metrics:
- TTC error
- false positive collision warnings
- false negative collision warnings
- time-to-warning
- lane-aware risk accuracy

---

## Research Roadmap

**Phase 1 — Sensor & Perception Foundation**
- [x] nuScenes playback
- [x] Camera pipeline
- [x] LiDAR pipeline
- [x] Semantic segmentation
- [x] Road segmentation
- [x] Classical lane detection

**Phase 2 — Temporal Perception**
- [x] Temporal lane tracking
- [x] Lane geometry estimation
- [x] Classical optical flow
- [x] RAFT-Small integration
- [ ] Quantitative optical-flow benchmarking
- [ ] Ego-motion estimation from optical flow
- [ ] Robust motion segmentation

**Phase 3 — 3D Perception**
- [x] LiDAR clustering baseline
- [x] Ground-truth detection interface
- [ ] PointPillars
- [ ] CenterPoint
- [ ] 3D detection benchmarking

**Phase 4 — Tracking & State Estimation**
- [x] Kalman filtering
- [x] Hungarian association
- [x] Track management
- [x] Velocity estimation
- [ ] Improved ego-state estimation
- [ ] Uncertainty propagation

**Phase 5 — Sensor Fusion**
- [ ] Camera-LiDAR association
- [ ] 2D-3D correspondence
- [ ] Fused object representation
- [ ] Multi-sensor tracking
- [ ] Temporal sensor fusion

**Phase 6 — Risk & Planning**
- [x] TTC foundation
- [ ] Lane-aware TTC
- [ ] Collision probability
- [ ] Trajectory prediction
- [ ] Risk-aware planning interface
- [ ] Autonomous Emergency Braking prototype

**Phase 7 — Closed-Loop Evaluation**
- [ ] CARLA integration
- [ ] Closed-loop perception
- [ ] Planning
- [ ] Control
- [ ] End-to-end scenario evaluation

---

## Design Principles

**Modular ROS 2 Architecture**

Each major algorithm is implemented as an independent ROS 2 node with explicit topic interfaces. This allows:

- independent debugging
- component replacement
- algorithm benchmarking
- easier experimentation
- clear system boundaries

**Detector Agnostic**

The tracking system is designed independently of the detection backend.

```
Ground Truth
DBSCAN LiDAR
PointPillars
CenterPoint
Camera-LiDAR Fusion
         │
         ▼
Object Detection Interface
         │
         ▼
      Tracker
```

**Classical + Learned Baselines**

Where appropriate, the project implements both classical and learned approaches, e.g. `Lucas-Kanade` vs `RAFT-Small`, and eventually `DBSCAN` vs `PointPillars / CenterPoint`. This makes computational and algorithmic trade-offs measurable.

**Temporal Reasoning**

The system explicitly incorporates temporal information through:

- lane tracking
- optical flow
- multi-object tracking
- velocity estimation
- TTC

The goal is to progress from independent frame-level perception toward temporally consistent scene understanding.

**Reproducibility**

The system exposes intermediate ROS 2 topics so that individual modules can be inspected independently:

```
Camera → Segmentation → Road Mask → Lane Detection
```

Each intermediate result can be visualized and debugged separately.

---

## Engineering Highlights

This repository demonstrates practical experience with:

- ROS 2 node architecture
- ROS 2 publishers/subscribers
- QoS configuration
- TF2
- RViz2
- camera perception
- LiDAR perception
- semantic segmentation
- classical computer vision
- optical flow
- deep-learning inference
- temporal filtering
- Kalman filtering
- Hungarian data association
- multi-object tracking
- velocity estimation
- collision-risk reasoning
- modular perception architecture
- dataset-driven autonomous-driving experimentation

---

## Why This Project?

Modern ADAS systems are not simply a single neural network. A practical autonomous-driving system contains multiple interconnected layers:

```
Sensors → Perception → State Estimation → Tracking → Prediction → Risk Reasoning → Planning → Control
```

This project focuses primarily on the **perception, temporal estimation, tracking, and collision-risk layers** that connect raw sensor measurements to autonomous-driving decisions.

The architecture is deliberately designed so that individual algorithms can be implemented, inspected, benchmarked, replaced, and extended — rather than hiding the complete pipeline behind a single end-to-end model.

---

## Future Direction

The long-term objective is to evolve this repository toward a research-oriented ADAS stack combining:

Camera + LiDAR + Semantic Understanding + Lane Geometry + Temporal Perception + 3D Detection + Multi-Object Tracking + Ego-State Estimation + Trajectory Prediction + Collision Risk + Planning

The final system should support quantitative evaluation of the complete perception-to-risk pipeline under realistic autonomous-driving scenarios.

---

## Technologies

**Robotics**
- ROS 2 Jazzy
- RViz2
- TF2
- nuScenes

**Computer Vision**
- OpenCV
- Shi-Tomasi
- Lucas-Kanade Optical Flow
- Canny Edge Detection
- Hough Transform
- Semantic Segmentation
- Lane Detection
- RAFT Optical Flow

**Machine Learning**
- PyTorch
- Torchvision
- SegFormer
- RAFT-Small

**LiDAR / 3D Perception**
- NumPy
- SciPy
- scikit-learn
- DBSCAN
- PointPillars
- CenterPoint

**State Estimation**
- Kalman Filtering
- Hungarian Assignment
- Multi-Object Tracking
- Ego-relative velocity estimation
- Time-To-Collision

---

## Author

**Gaurav Ramteke**

Robotics | Computer Vision | ADAS Perception | Autonomous Systems

GitHub: https://github.com/GauravR2012

---

## License

This project is intended primarily for research, experimentation, and educational purposes.

Add an explicit license here if/when the repository is released under a specific open-source license.
