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
        # CONFIGURATION
        # ============================================================

        self.model_name = (
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )

        self.device = torch.device("cpu")

        # CPU threading.
        # We will benchmark this later.
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
        # ROS QoS
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

        self.overlay_pub = self.create_publisher(
            Image,
            "/camera/segmentation/overlay",
            qos,
        )

        self.mask_pub = self.create_publisher(
            Image,
            "/camera/segmentation/mask",
            qos,
        )

        # ============================================================
        # FRAME BUFFER
        # ============================================================

        self.frame_lock = threading.Lock()

        self.latest_msg = None

        # Event used to wake the worker when a new frame arrives.
        self.frame_event = threading.Event()

        # ============================================================
        # RESULT BUFFER
        # ============================================================

        self.result_lock = threading.Lock()

        self.latest_overlay = None
        self.latest_mask = None

        # ============================================================
        # STATISTICS
        # ============================================================

        self.frames_received = 0
        self.frames_processed = 0

        self.last_inference_time = 0.0

        self.last_stats_time = time.time()

        self.stats_processed = 0

        # ============================================================
        # LOAD SEGFORMER
        # ============================================================

        self.get_logger().info(
            "Loading SegFormer..."
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
            "SegFormer loaded successfully."
        )

        self.num_classes = (
            self.model.config.num_labels
        )

        self.get_logger().info(
            f"Segmentation classes: "
            f"{self.num_classes}"
        )

        # ============================================================
        # COLOR PALETTE
        # ============================================================

        self.palette = self.create_palette(
            self.num_classes
        )

        # ============================================================
        # ROS PUBLISH TIMER
        # ============================================================

        # IMPORTANT:
        #
        # This timer ONLY publishes already-computed results.
        #
        # It never performs neural-network inference.

        self.publish_timer = self.create_timer(
            0.05,
            self.publish_result,
        )

        # ============================================================
        # WORKER CONTROL
        # ============================================================

        self.shutdown_event = threading.Event()

        self.worker_thread = threading.Thread(
            target=self.segmentation_worker,
            daemon=True,
        )

        self.worker_thread.start()

        self.get_logger().info(
            "Segmentation worker started."
        )

        self.get_logger().info(
            "Waiting for /camera/front/image ..."
        )

    # ================================================================
    # CAMERA CALLBACK
    # ================================================================

    def image_callback(self, msg):

        # ------------------------------------------------------------
        # This callback MUST remain lightweight.
        #
        # No OpenCV processing.
        # No PyTorch.
        # No SegFormer.
        # ------------------------------------------------------------

        with self.frame_lock:

            # Always retain only the newest frame.
            self.latest_msg = msg

        self.frames_received += 1

        # Wake segmentation worker.
        self.frame_event.set()

    # ================================================================
    # SEGMENTATION WORKER
    # ================================================================

    def segmentation_worker(self):

        self.get_logger().info(
            "Segmentation worker thread running."
        )

        while not self.shutdown_event.is_set():

            # --------------------------------------------------------
            # Wait for a camera frame
            # --------------------------------------------------------

            self.frame_event.wait()

            if self.shutdown_event.is_set():
                break

            # --------------------------------------------------------
            # Get newest frame
            # --------------------------------------------------------

            with self.frame_lock:

                msg = self.latest_msg

                self.latest_msg = None

                # Clear the event BEFORE inference.
                #
                # If another camera frame arrives while inference
                # is running, image_callback() will set the event
                # again.
                self.frame_event.clear()

            if msg is None:
                continue

            # --------------------------------------------------------
            # Convert ROS image → OpenCV
            # --------------------------------------------------------

            try:

                image_bgr = self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding="bgr8",
                )

            except Exception as e:

                self.get_logger().error(
                    f"CvBridge conversion failed: {e}"
                )

                continue

            # --------------------------------------------------------
            # BGR → RGB
            # --------------------------------------------------------

            image_rgb = cv2.cvtColor(
                image_bgr,
                cv2.COLOR_BGR2RGB,
            )

            original_height, original_width = (
                image_rgb.shape[:2]
            )

            # ========================================================
            # SEGFORMER PREPROCESSING
            # ========================================================

            inputs = self.processor(
                images=image_rgb,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            # ========================================================
            # INFERENCE
            # ========================================================

            start_time = time.time()

            with torch.inference_mode():

                outputs = self.model(
                    **inputs
                )

            inference_time = (
                time.time() - start_time
            )

            # ========================================================
            # RESIZE SEGMENTATION LOGITS
            # ========================================================

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

            # ========================================================
            # CLASS PREDICTION
            # ========================================================

            segmentation = (
                logits
                .argmax(dim=1)
                [0]
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            # ========================================================
            # COLOR MASK
            # ========================================================

            color_mask_rgb = self.palette[
                segmentation
            ]

            color_mask_bgr = cv2.cvtColor(
                color_mask_rgb,
                cv2.COLOR_RGB2BGR,
            )

            # ========================================================
            # OVERLAY
            # ========================================================

            overlay = cv2.addWeighted(
                image_bgr,
                0.55,
                color_mask_bgr,
                0.45,
                0.0,
            )

            # ========================================================
            # CONVERT RESULTS TO ROS MESSAGES
            # ========================================================

            overlay_msg = (
                self.bridge.cv2_to_imgmsg(
                    overlay,
                    encoding="bgr8",
                )
            )

            overlay_msg.header = msg.header

            mask_msg = (
                self.bridge.cv2_to_imgmsg(
                    segmentation,
                    encoding="mono8",
                )
            )

            mask_msg.header = msg.header

            # ========================================================
            # STORE RESULT
            # ========================================================

            with self.result_lock:

                self.latest_overlay = overlay_msg

                self.latest_mask = mask_msg

                self.last_inference_time = (
                    inference_time
                )

            self.frames_processed += 1

            self.stats_processed += 1

            # ========================================================
            # LOGGING
            # ========================================================

            now = time.time()

            if (
                now - self.last_stats_time
                >= 5.0
            ):

                elapsed = (
                    now - self.last_stats_time
                )

                processing_fps = (
                    self.stats_processed
                    / max(elapsed, 1e-6)
                )

                self.get_logger().info(
                    f"Segmentation | "
                    f"inference="
                    f"{inference_time:.2f}s | "
                    f"worker FPS="
                    f"{processing_fps:.2f} | "
                    f"received="
                    f"{self.frames_received} | "
                    f"processed="
                    f"{self.frames_processed}"
                )

                self.stats_processed = 0

                self.last_stats_time = now

    # ================================================================
    # PUBLISH RESULT
    # ================================================================

    def publish_result(self):

        # ------------------------------------------------------------
        # Only a very small ROS operation occurs here.
        #
        # There is NO model inference.
        # ------------------------------------------------------------

        with self.result_lock:

            overlay = self.latest_overlay

            mask = self.latest_mask

            # Clear buffers after taking the result.
            self.latest_overlay = None
            self.latest_mask = None

        if overlay is not None:

            self.overlay_pub.publish(
                overlay
            )

        if mask is not None:

            self.mask_pub.publish(
                mask
            )

    # ================================================================
    # COLOR PALETTE
    # ================================================================

    @staticmethod
    def create_palette(
        num_classes
    ):

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

    # ================================================================
    # SHUTDOWN
    # ================================================================

    def shutdown_worker(self):

        self.get_logger().info(
            "Stopping segmentation worker..."
        )

        self.shutdown_event.set()

        # Wake worker if it is currently waiting.
        self.frame_event.set()

        if self.worker_thread.is_alive():

            self.worker_thread.join(
                timeout=5.0
            )

        self.get_logger().info(
            "Segmentation worker stopped."
        )


# ====================================================================
# MAIN
# ====================================================================

def main(args=None):

    rclpy.init(args=args)

    node = SemanticSegmentationNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        node.get_logger().info(
            "Keyboard interrupt."
        )

    finally:

        node.shutdown_worker()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()
