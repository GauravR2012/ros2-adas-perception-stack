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
