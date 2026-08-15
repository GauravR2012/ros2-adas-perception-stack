import os
import time

import cv2
import numpy as np
import torch

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

IMAGE_PATH = (
    "/home/adarsh/av_perception/data/nuscenes/"
    "samples/CAM_FRONT/"
)

OUTPUT_PATH = (
    "/home/adarsh/ros2_ws/"
    "segmentation_test.png"
)


# ============================================================
# FIND ONE CAMERA IMAGE
# ============================================================

files = sorted(
    [
        os.path.join(IMAGE_PATH, f)
        for f in os.listdir(IMAGE_PATH)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]
)

if len(files) == 0:
    raise RuntimeError(
        f"No images found in {IMAGE_PATH}"
    )

image_path = files[0]

print("Image:")
print(image_path)


# ============================================================
# LOAD IMAGE
# ============================================================

image_bgr = cv2.imread(
    image_path
)

if image_bgr is None:
    raise RuntimeError(
        f"Could not read {image_path}"
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
# LOAD MODEL
# ============================================================

print()
print("Loading SegFormer...")

processor = (
    SegformerImageProcessor.from_pretrained(
        MODEL_NAME
    )
)

model = (
    SegformerForSemanticSegmentation
    .from_pretrained(
        MODEL_NAME
    )
)

device = torch.device("cpu")

model.to(device)
model.eval()

print("Model loaded.")
print("Device:", device)


# ============================================================
# PREPROCESS
# ============================================================

inputs = processor(
    images=image_rgb,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# ============================================================
# INFERENCE
# ============================================================

print()
print("Running segmentation...")

start = time.time()

with torch.no_grad():

    outputs = model(
        **inputs
    )

elapsed = (
    time.time() - start
)

print(
    f"Inference time: {elapsed:.2f} s"
)


# ============================================================
# GET SEGMENTATION MAP
# ============================================================

logits = outputs.logits

print(
    "Logits shape:",
    tuple(logits.shape)
)

# SegFormer logits are lower resolution than the
# original image. Resize them back to image resolution.

logits = torch.nn.functional.interpolate(
    logits,
    size=image_rgb.shape[:2],
    mode="bilinear",
    align_corners=False
)

segmentation = (
    logits.argmax(
        dim=1
    )[0]
    .cpu()
    .numpy()
)

print(
    "Segmentation shape:",
    segmentation.shape
)

print(
    "Number of predicted classes:",
    len(np.unique(segmentation))
)


# ============================================================
# CREATE COLOR MASK
# ============================================================

num_classes = model.config.num_labels

rng = np.random.default_rng(42)

palette = rng.integers(
    0,
    255,
    size=(num_classes, 3),
    dtype=np.uint8
)

mask_rgb = palette[
    segmentation
]

mask_bgr = cv2.cvtColor(
    mask_rgb,
    cv2.COLOR_RGB2BGR
)


# ============================================================
# OVERLAY
# ============================================================

overlay = cv2.addWeighted(
    image_bgr,
    0.55,
    mask_bgr,
    0.45,
    0
)


# ============================================================
# SAVE
# ============================================================

cv2.imwrite(
    OUTPUT_PATH,
    overlay
)

print()
print("Saved:")
print(OUTPUT_PATH)
