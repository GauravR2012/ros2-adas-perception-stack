import os
import torch
import numpy as np

from mmengine.config import Config
from mmdet3d.apis import init_model, inference_detector
from nuscenes.nuscenes import NuScenes

# -----------------------------------------
# CONFIG
# -----------------------------------------
NUSC_ROOT = "/home/adarsh/av_perception/data/nuscenes"
VERSION = "v1.0-mini"

CONFIG_FILE = "/home/adarsh/mmdetection3d/configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py"
CHECKPOINT = "/home/adarsh/mmdetection3d/checkpoints/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d_20210818_153306-7a09c533.pth"

DEVICE = "cpu"

# -----------------------------------------
# LOAD MODEL
# -----------------------------------------
print("Loading PointPillars model...")

model = init_model(CONFIG_FILE, CHECKPOINT, device=DEVICE)

print("Model loaded successfully.")

# -----------------------------------------
# LOAD NUSCENES
# -----------------------------------------
nusc = NuScenes(version=VERSION, dataroot=NUSC_ROOT, verbose=False)

scene = nusc.scene[0]
sample_token = scene["first_sample_token"]
sample = nusc.get("sample", sample_token)

lidar_token = sample["data"]["LIDAR_TOP"]
lidar_data = nusc.get("sample_data", lidar_token)

lidar_path = os.path.join(NUSC_ROOT, lidar_data["filename"])

print("Running inference...")

result, data = inference_detector(model, lidar_path)

boxes = result.pred_instances_3d.bboxes_3d
scores = result.pred_instances_3d.scores_3d
labels = result.pred_instances_3d.labels_3d

print(f"\nDetected {len(boxes)} objects\n")

for i in range(len(boxes)):
    print("Object", i)
    print("Center:", boxes[i].center.numpy())
    print("Dims:", boxes[i].dims.numpy())
    print("Yaw:", boxes[i].yaw.numpy())
    print("Score:", float(scores[i]))
    print("-" * 40)
