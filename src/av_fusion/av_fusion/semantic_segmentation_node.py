#!/usr/bin/env python3

import threading
import time

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

        # ==========================================================
        # CONFIGURATION
        # ==========================================================

        self.model_name = (
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        )

        self.device = torch.device("cpu")

        # ==========================================================
        # LOGGING
        # ==========================================================

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

        # ==========================================================
        # CV BRIDGE
        # ==========================================================

        self.bridge = CvBridge()

        # ==========================================================
        # ROS QoS
        # ==========================================================

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        # ==========================================================
        # SUBSCRIBER
        # ==========================================================

        self.image_sub = self.create_subscription(
            Image,
            "/camera/front/image",
            self.image_callback,
            qos
        )

        # ==========================================================
        # PUBLISHERS
        # ==========================================================

        # Raw semantic class-ID image.
        #
        # Encoding:
        #
        # 0  = road
        # 1  = sidewalk
        # ...
        # 13 = car
        # 14 = truck
        # 15 = bus
        #
        self.mask_pub = self.create_publisher(
            Image,
            "/camera/segmentation/mask",
            qos
        )

        # Human/RViz-friendly visualization.
        self.overlay_pub = self.create_publisher(
            Image,
            "/camera/segmentation/overlay",
            qos
        )

        # ==========================================================
        # LOAD SEGFORMER
        # ==========================================================

        self.get_logger().info(
            "Loading Cityscapes SegFormer..."
        )

        self.processor = (
            SegformerImageProcessor.from_pretrained(
                self.model_name
            )
        )

        self.model = (
            SegformerForSemanticSegmentation.from_pretrained(
                self.model_name
            )
        )

        self.model.to(self.device)
        self.model.eval()

        self.get_logger().info(
            "SegFormer loaded successfully."
        )

        self.get_logger().info(
            f"Segmentation classes: "
            f"{self.model.config.num_labels}"
        )

        # ==========================================================
        # PRINT CLASS MAP
        # ==========================================================

        self.id2label = {
            int(k): v
            for k, v in self.model.config.id2label.items()
        }

        self.get_logger().info(
            "Cityscapes class mapping:"
        )

        for class_id in sorted(self.id2label.keys()):

            self.get_logger().info(
                f"  {class_id:2d}: "
                f"{self.id2label[class_id]}"
            )

        # ==========================================================
        # LATEST-FRAME BUFFER
        # ==========================================================

        self.latest_image = None

        self.latest_header = None

        self.frame_lock = threading.Lock()

        self.new_frame_event = threading.Event()

        # ==========================================================
        # STATISTICS
        # ==========================================================

        self.received_frames = 0
        self.processed_frames = 0

        self.last_fps_time = time.time()

        # ==========================================================
        # WORKER THREAD
        # ==========================================================

        self.worker_running = True

        self.worker_thread = threading.Thread(
            target=self.segmentation_worker,
            daemon=True
        )

        self.worker_thread.start()

        self.get_logger().info(
            "Segmentation worker started."
        )

        self.get_logger().info(
            "Segmentation worker thread running."
        )

        self.get_logger().info(
            "Waiting for /camera/front/image ..."
        )

    # ==============================================================
    # IMAGE CALLBACK
    # ==============================================================

    def image_callback(self, msg):

        self.received_frames += 1

        try:

            image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

        except Exception as e:

            self.get_logger().error(
                f"CV bridge conversion failed: {e}"
            )

            return

        # ----------------------------------------------------------
        # IMPORTANT:
        #
        # Keep ONLY the newest frame.
        #
        # We do not want the CPU worker processing old images while
        # the NuScenes player continues publishing newer ones.
        # ----------------------------------------------------------

        with self.frame_lock:

            self.latest_image = image

            self.latest_header = msg.header

        self.new_frame_event.set()

    # ==============================================================
    # SEGMENTATION WORKER
    # ==============================================================

    def segmentation_worker(self):

        while self.worker_running:

            # ------------------------------------------------------
            # Wait for a frame
            # ------------------------------------------------------

            self.new_frame_event.wait(
                timeout=1.0
            )

            if not self.worker_running:
                break

            # ------------------------------------------------------
            # Get newest frame
            # ------------------------------------------------------

            with self.frame_lock:

                if self.latest_image is None:

                    self.new_frame_event.clear()

                    continue

                image = self.latest_image.copy()

                header = self.latest_header

                # We are going to process this frame.
                #
                # Clear the event. If a newer frame arrives while
                # inference is running, the callback will set it
                # again.
                self.new_frame_event.clear()

            # ------------------------------------------------------
            # BGR → RGB
            # ------------------------------------------------------

            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            # ------------------------------------------------------
            # INFERENCE
            # ------------------------------------------------------

            start_time = time.time()

            try:

                inputs = self.processor(
                    images=image_rgb,
                    return_tensors="pt"
                )

                inputs = {
                    key: value.to(self.device)
                    for key, value in inputs.items()
                }

                with torch.no_grad():

                    outputs = self.model(
                        **inputs
                    )

            except Exception as e:

                self.get_logger().error(
                    f"Segmentation inference failed: {e}"
                )

                continue

            inference_time = (
                time.time() - start_time
            )

            # ======================================================
            # RESIZE LOGITS
            # ======================================================

            logits = outputs.logits

            logits = torch.nn.functional.interpolate(
                logits,
                size=image_rgb.shape[:2],
                mode="bilinear",
                align_corners=False
            )

            # ======================================================
            # CLASS PREDICTION
            # ======================================================

            segmentation = logits.argmax(
                dim=1
            )[0]

            segmentation = (
                segmentation
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            # ======================================================
            # PUBLISH RAW CLASS-ID MASK
            # ======================================================

            mask_msg = self.bridge.cv2_to_imgmsg(
                segmentation,
                encoding="mono8"
            )

            mask_msg.header = header

            self.mask_pub.publish(
                mask_msg
            )

            # ======================================================
            # CREATE COLORED VISUALIZATION
            # ======================================================

            overlay = self.create_overlay(
                image,
                segmentation
            )

            overlay_msg = self.bridge.cv2_to_imgmsg(
                overlay,
                encoding="bgr8"
            )

            overlay_msg.header = header

            self.overlay_pub.publish(
                overlay_msg
            )

            # ======================================================
            # STATISTICS
            # ======================================================

            self.processed_frames += 1

            elapsed_since_fps = (
                time.time() - self.last_fps_time
            )

            if elapsed_since_fps >= 5.0:

                worker_fps = (
                    self.processed_frames
                    / elapsed_since_fps
                )

                self.get_logger().info(
                    "Segmentation | "
                    f"inference={inference_time:.2f}s | "
                    f"worker FPS={worker_fps:.2f} | "
                    f"received={self.received_frames} | "
                    f"processed={self.processed_frames}"
                )

                self.processed_frames = 0

                self.last_fps_time = time.time()

    # ==============================================================
    # VISUALIZATION
    # ==============================================================

    def create_overlay(
        self,
        image,
        segmentation
    ):

        # ----------------------------------------------------------
        # Cityscapes color palette.
        #
        # RGB colors.
        # ----------------------------------------------------------

        palette = np.array(
            [
                [128,  64, 128],   # 0 road
                [244,  35, 232],   # 1 sidewalk
                [ 70,  70,  70],   # 2 building
                [102, 102, 156],   # 3 wall
                [190, 153, 153],   # 4 fence
                [153, 153, 153],   # 5 pole
                [250, 170,  30],   # 6 traffic light
                [220, 220,   0],   # 7 traffic sign
                [107, 142,  35],   # 8 vegetation
                [152, 251, 152],   # 9 terrain
                [ 70, 130, 180],   # 10 sky
                [220,  20,  60],   # 11 person
                [255,   0,   0],   # 12 rider
                [  0,   0, 142],   # 13 car
                [  0,   0,  70],   # 14 truck
                [  0,  60, 100],   # 15 bus
                [  0,  80, 100],   # 16 train
                [  0,   0, 230],   # 17 motorcycle
                [119,  11,  32],   # 18 bicycle
            ],
            dtype=np.uint8
        )

        # ----------------------------------------------------------
        # Convert class IDs → RGB colors
        # ----------------------------------------------------------

        segmentation_color = palette[
            segmentation
        ]

        # ----------------------------------------------------------
        # RGB → BGR for OpenCV
        # ----------------------------------------------------------

        segmentation_bgr = cv2.cvtColor(
            segmentation_color,
            cv2.COLOR_RGB2BGR
        )

        # ----------------------------------------------------------
        # Overlay
        # ----------------------------------------------------------

        overlay = cv2.addWeighted(
            image,
            0.55,
            segmentation_bgr,
            0.45,
            0
        )

        return overlay

    # ==============================================================
    # SHUTDOWN
    # ==============================================================

    def destroy_node(self):

        self.get_logger().info(
            "Stopping segmentation worker..."
        )

        self.worker_running = False

        self.new_frame_event.set()

        if self.worker_thread.is_alive():

            self.worker_thread.join(
                timeout=2.0
            )

        super().destroy_node()


# ==================================================================
# MAIN
# ==================================================================

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