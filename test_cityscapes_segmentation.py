import time
import torch
import cv2
import numpy as np

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)

MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

print("=" * 60)
print("Cityscapes SegFormer Test")
print("=" * 60)

print("Loading processor...")
processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)

print("Loading model...")
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)

device = torch.device("cpu")
model.to(device)
model.eval()

print("Model loaded.")
print("Device:", device)

print("Number of classes:", model.config.num_labels)

print("\nClasses:")
for idx, name in model.config.id2label.items():
    print(f"{idx:2d}: {name}")

print("=" * 60)

