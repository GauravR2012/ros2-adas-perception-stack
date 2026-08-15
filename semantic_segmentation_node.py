#!/usr/bin/env python3

import time
import threading

import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
)

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)


class SemanticSegmentationNode(Node):

    def __init__(self):

        super().__init__("semantic_segmentation_node")

        # ============================================================
        # CONFIG
        # ============================================================

        self.model_name = (
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )

        self.device = torch.device("cpu")

        # Limit CPU usage somewhat.
        # Change later after benchmarking.
        torch.set_num_threads(4)

        self.get_logger().info(
            "=========================================="
        )
        self.get_logger().info(
            "Semantic Segmentation Node"
        )
        self.get_logger().info(
            "=========================================="
        )
        self.get_logger().info(
            f"Model: {self.model_name}"
        )
        self.get_logger().info(
            f"Device: {self.device}"
        )

        # ============================================================
        # CV BRIDGE
        # ============================================================

        self.bridge = CvBridge()

        # ============================================================
        # QOS
        # ============================================================

        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )

        # ============================================================
        # SUBSCRIBER
        # ============================================================

        self.image_sub = self.create_subscription(
            Image,
            "/camera/front/image",
            self.image_callback,
            qos,
        )

        # ============================================================
        # PUBLISHERS
        # ============================================================

        # Colored segmentation overlay
        self.overlay_pub = self.create_publisher(
            Image,
            "/camera/segmentation/overlay",
            qos,
        )

        # Raw semantic class-ID image
        self.mask_pub = self.create_publisher(
            Image,
            "/camera/segmentation/mask",
            qos,
        )

        # ============================================================
        # LATEST FRAME BUFFER
        # ============================================================

        self.latest_msg = None

        self.frame_lock = threading.Lock()

        self.frames_received = 0
        self.frames_processed = 0

        # ============================================================
        # LOAD MODEL
        # ============================================================

        self.get_logger().info(
            "Loading SegFormer model..."
        )

        self.processor = (
            SegformerImageProcessor.from_pretrained(
                self.model_name
            )
        )

        self.model = (
            SegformerForSemanticSegmentation
            .from_pretrained(
                self.model_name
            )
        )

        self.model.to(self.device)
        self.model.eval()

        self.get_logger().info(
            "SegFormer model loaded successfully."
        )

        # ============================================================
        # COLOR PALETTE
        # ============================================================

        self.palette = self.create_palette(
            self.model.config.num_labels
        )

        self.get_logger().info(
            f"Number of segmentation classes: "
            f"{self.model.config.num_labels}"
        )

        # ============================================================
        # INFERENCE TIMER
        # ============================================================

        # We don't run inference directly inside the image callback.
        #
        # The callback only stores the newest frame.
        #
        # This prevents the segmentation model from blocking
        # reception of camera messages.

        self.inference_timer = self.create_timer(
            0.01,
            self.process_latest_frame,
        )

        # ============================================================
        # STATS
        # ============================================================

        self.last_log_time = time.time()

        self.get_logger().info(
            "Waiting for /camera/front/image ..."
        )

    # ================================================================
    # IMAGE CALLBACK
    # ================================================================

    def image_callback(self, msg):

        with self.frame_lock:

            self.latest_msg = msg

        self.frames_received += 1

    # ================================================================
    # PROCESS LATEST FRAME
    # ================================================================

    def process_latest_frame(self):

        # ------------------------------------------------------------
        # Get latest image
        # ------------------------------------------------------------

        with self.frame_lock:

            if self.latest_msg is None:
                return

            msg = self.latest_msg

            # Clear it so that we don't repeatedly process
            # exactly the same frame.
            self.latest_msg = None

        # ------------------------------------------------------------
        # ROS Image → OpenCV
        # ------------------------------------------------------------

        try:

            image_bgr = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )

        except Exception as e:

            self.get_logger().error(
                f"CvBridge conversion failed: {e}"
            )

            return

        # ------------------------------------------------------------
        # BGR → RGB
        # ------------------------------------------------------------

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        original_height, original_width = (
            image_rgb.shape[:2]
        )

        # ------------------------------------------------------------
        # SegFormer preprocessing
        # ------------------------------------------------------------

        inputs = self.processor(
            images=image_rgb,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # ------------------------------------------------------------
        # INFERENCE
        # ------------------------------------------------------------

        start_time = time.time()

        with torch.no_grad():

            outputs = self.model(
                **inputs
            )

        inference_time = (
            time.time() - start_time
        )

        # ------------------------------------------------------------
        # Resize logits to original image size
        # ------------------------------------------------------------

        logits = outputs.logits

        logits = torch.nn.functional.interpolate(
            logits,
            size=(
                original_height,
                original_width,
            ),
            mode="bilinear",
            align_corners=False,
        )

        # ------------------------------------------------------------
        # Class prediction
        # ------------------------------------------------------------

        segmentation = (
            logits
            .argmax(dim=1)
            [0]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        # ============================================================
        # CREATE COLOR MASK
        # ============================================================

        color_mask_rgb = self.palette[
            segmentation
        ]

        color_mask_bgr = cv2.cvtColor(
            color_mask_rgb,
            cv2.COLOR_RGB2BGR,
        )

        # ============================================================
        # CREATE OVERLAY
        # ============================================================

        overlay = cv2.addWeighted(
            image_bgr,
            0.55,
            color_mask_bgr,
            0.45,
            0.0,
        )

        # ============================================================
        # PUBLISH OVERLAY
        # ============================================================

        overlay_msg = self.bridge.cv2_to_imgmsg(
            overlay,
            encoding="bgr8",
        )

        overlay_msg.header = msg.header

        self.overlay_pub.publish(
            overlay_msg
        )

        # ============================================================
        # PUBLISH RAW MASK
        # ============================================================

        mask_msg = self.bridge.cv2_to_imgmsg(
            segmentation,
            encoding="mono8",
        )

        mask_msg.header = msg.header

        self.mask_pub.publish(
            mask_msg
        )

        # ============================================================
        # STATISTICS
        # ============================================================

        self.frames_processed += 1

        now = time.time()

        if now - self.last_log_time >= 2.0:

            processing_fps = (
                self.frames_processed
                / max(
                    now - self.last_log_time,
                    1e-6,
                )
            )

            self.get_logger().info(
                f"Segmentation | "
                f"inference={inference_time:.2f}s | "
                f"latest FPS={1.0 / max(inference_time, 1e-6):.2f}"
            )

            self.last_log_time = now

            self.frames_processed = 0

    # ================================================================
    # CREATE PALETTE
    # ================================================================

    @staticmethod
    def create_palette(num_classes):

        rng = np.random.default_rng(
            seed=42
        )

        palette = rng.integers(
            low=0,
            high=256,
            size=(
                num_classes,
                3,
            ),
            dtype=np.uint8,
        )

        return palette


# ====================================================================
# MAIN
# ====================================================================

def main(args=None):

    rclpy.init(args=args)

    node = SemanticSegmentationNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()
