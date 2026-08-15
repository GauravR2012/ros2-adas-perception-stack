import os
import time

import cv2
import numpy as np
import torch

from nuscenes.nuscenes import NuScenes
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)


# ============================================================
# CONFIG
# ============================================================

NUSCENES_ROOT = "/home/adarsh/av_perception/data/nuscenes"
MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

OUTPUT = "/home/adarsh/ros2_ws/cityscapes_segmentation_result.jpg"


# ============================================================
# LOAD NUSCENES
# ============================================================

print("Loading NuScenes...")

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=NUSCENES_ROOT,
    verbose=False
)

scene = nusc.scene[0]

sample_token = scene["first_sample_token"]

sample = nusc.get(
    "sample",
    sample_token
)

cam_token = sample["data"]["CAM_FRONT"]

cam_data = nusc.get(
    "sample_data",
    cam_token
)

image_path = os.path.join(
    NUSCENES_ROOT,
    cam_data["filename"]
)

print("Image:")
print(image_path)


# ============================================================
# LOAD IMAGE
# ============================================================

image_bgr = cv2.imread(image_path)

if image_bgr is None:
    raise RuntimeError(
        f"Could not load image: {image_path}"
    )

image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)

print(
    "Image shape:",
    image_rgb.shape
)


# ============================================================
# LOAD SEGFORMER
# ============================================================

print("\nLoading Cityscapes SegFormer...")

processor = SegformerImageProcessor.from_pretrained(
    MODEL_NAME
)

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME
)

device = torch.device("cpu")

model.to(device)
model.eval()

print("Model ready.")


# ============================================================
# INFERENCE
# ============================================================

print("\nRunning inference...")

start = time.time()

inputs = processor(
    images=image_rgb,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

with torch.no_grad():

    outputs = model(**inputs)

inference_time = time.time() - start

print(
    f"Inference time: {inference_time:.2f} s"
)


# ============================================================
# SEGMENTATION MASK
# ============================================================

logits = outputs.logits

print(
    "Logits shape:",
    tuple(logits.shape)
)

# Resize logits to original image size
logits = torch.nn.functional.interpolate(
    logits,
    size=image_rgb.shape[:2],
    mode="bilinear",
    align_corners=False
)

segmentation = logits.argmax(
    dim=1
)[0]

segmentation = segmentation.cpu().numpy()

print(
    "Segmentation shape:",
    segmentation.shape
)

print(
    "Unique classes:",
    np.unique(segmentation)
)


# ============================================================
# CLASS MAP
# ============================================================

id2label = model.config.id2label

print("\nDetected classes:")

unique_ids, counts = np.unique(
    segmentation,
    return_counts=True
)

for class_id, count in zip(
    unique_ids,
    counts
):

    class_name = id2label[int(class_id)]

    percentage = (
        count /
        segmentation.size *
        100
    )

    print(
        f"{int(class_id):2d}: "
        f"{class_name:15s} "
        f"{percentage:6.2f}%"
    )


# ============================================================
# VISUALIZATION
# ============================================================

# Simple deterministic palette.
# Each Cityscapes class receives a different color.

palette = np.array([
    [128,  64, 128],   # road
    [244,  35, 232],   # sidewalk
    [ 70,  70,  70],   # building
    [102, 102, 156],   # wall
    [190, 153, 153],   # fence
    [153, 153, 153],   # pole
    [250, 170,  30],   # traffic light
    [220, 220,   0],   # traffic sign
    [107, 142,  35],   # vegetation
    [152, 251, 152],   # terrain
    [ 70, 130, 180],   # sky
    [220,  20,  60],   # person
    [255,   0,   0],   # rider
    [  0,   0, 142],   # car
    [  0,   0,  70],   # truck
    [  0,  60, 100],   # bus
    [  0,  80, 100],   # train
    [  0,   0, 230],   # motorcycle
    [119,  11,  32],   # bicycle
], dtype=np.uint8)


mask_color = palette[
    segmentation
]

mask_color_bgr = cv2.cvtColor(
    mask_color,
    cv2.COLOR_RGB2BGR
)


# ============================================================
# OVERLAY
# ============================================================

overlay = cv2.addWeighted(
    image_bgr,
    0.55,
    mask_color_bgr,
    0.45,
    0
)


# ============================================================
# SAVE
# ============================================================

cv2.imwrite(
    OUTPUT,
    overlay
)

print("\nSaved:")
print(OUTPUT)

print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)
