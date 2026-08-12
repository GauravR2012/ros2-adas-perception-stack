#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy
)

from sensor_msgs.msg import Image

from cv_bridge import CvBridge

import numpy as np
import cv2


class RoadSegmentationNode(Node):

    def __init__(self):

        super().__init__("road_segmentation_node")

        # ==========================================================
        # CONFIGURATION
        # ==========================================================

        # Cityscapes SegFormer class ID
        #
        # 0 = road
        #
        self.ROAD_CLASS_ID = 0

        # ==========================================================
        # STARTUP LOG
        # ==========================================================

        self.get_logger().info(
            "=========================================="
        )

        self.get_logger().info(
            "Road Segmentation Node"
        )

        self.get_logger().info(
            "=========================================="
        )

        self.get_logger().info(
            "Input: /camera/segmentation/mask"
        )

        self.get_logger().info(
            "Road class ID: 0"
        )

        # ==========================================================
        # CV BRIDGE
        # ==========================================================

        self.bridge = CvBridge()

        # ==========================================================
        # QoS
        # ==========================================================

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        # ==========================================================
        # SUBSCRIBER
        # ==========================================================

        self.mask_sub = self.create_subscription(
            Image,
            "/camera/segmentation/mask",
            self.mask_callback,
            qos
        )

        # ==========================================================
        # PUBLISHERS
        # ==========================================================

        # Binary road mask
        self.road_mask_pub = self.create_publisher(
            Image,
            "/camera/segmentation/road_mask",
            qos
        )

        # Visualization
        self.road_overlay_pub = self.create_publisher(
            Image,
            "/camera/segmentation/road_overlay",
            qos
        )

        # ==========================================================
        # STATISTICS
        # ==========================================================

        self.processed_frames = 0

        self.get_logger().info(
            "Road segmentation node started."
        )

        self.get_logger().info(
            "Waiting for segmentation mask..."
        )

    # ==============================================================
    # CALLBACK
    # ==============================================================

    def mask_callback(self, msg):

        try:

            # ------------------------------------------------------
            # ROS Image → NumPy
            # ------------------------------------------------------

            segmentation = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="mono8"
            )

        except Exception as e:

            self.get_logger().error(
                f"Failed to convert segmentation mask: {e}"
            )

            return

        # ----------------------------------------------------------
        # Extract road
        #
        # SegFormer:
        #
        # 0 = road
        #
        # Output:
        #
        # 255 = road
        #   0 = non-road
        # ----------------------------------------------------------

        road_mask = np.where(
            segmentation == self.ROAD_CLASS_ID,
            255,
            0
        ).astype(np.uint8)

        # ==========================================================
        # PUBLISH BINARY ROAD MASK
        # ==========================================================

        road_mask_msg = self.bridge.cv2_to_imgmsg(
            road_mask,
            encoding="mono8"
        )

        road_mask_msg.header = msg.header

        self.road_mask_pub.publish(
            road_mask_msg
        )

        # ==========================================================
        # CREATE VISUALIZATION
        # ==========================================================

        # Make road region white.
        road_visualization = np.zeros(
            (
                road_mask.shape[0],
                road_mask.shape[1],
                3
            ),
            dtype=np.uint8
        )

        road_visualization[
            road_mask == 255
        ] = [255, 255, 255]

        # ==========================================================
        # PUBLISH VISUALIZATION
        # ==========================================================

        overlay_msg = self.bridge.cv2_to_imgmsg(
            road_visualization,
            encoding="bgr8"
        )

        overlay_msg.header = msg.header

        self.road_overlay_pub.publish(
            overlay_msg
        )

        # ==========================================================
        # STATISTICS
        # ==========================================================

        self.processed_frames += 1

        road_pixels = np.count_nonzero(
            road_mask
        )

        total_pixels = road_mask.size

        road_percentage = (
            100.0
            * road_pixels
            / total_pixels
        )

        # Don't log every frame.
        if self.processed_frames % 10 == 0:

            self.get_logger().info(
                "Road segmentation | "
                f"frames={self.processed_frames} | "
                f"road={road_percentage:.1f}%"
            )


# ==================================================================
# MAIN
# ==================================================================

def main(args=None):

    rclpy.init(args=args)

    node = RoadSegmentationNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()
