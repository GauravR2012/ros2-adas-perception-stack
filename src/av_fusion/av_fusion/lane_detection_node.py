#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy
)

from sensor_msgs.msg import Image

from cv_bridge import CvBridge


class LaneDetectionNode(Node):

    def __init__(self):

        super().__init__("lane_detection_node")

        # ==========================================================
        # STARTUP
        # ==========================================================

        self.get_logger().info(
            "=========================================="
        )
        self.get_logger().info(
            "Lane Detection Node"
        )
        self.get_logger().info(
            "=========================================="
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
        # SUBSCRIBERS
        # ==========================================================

        self.image_sub = self.create_subscription(
            Image,
            "/camera/front/image",
            self.image_callback,
            qos
        )

        self.road_mask_sub = self.create_subscription(
            Image,
            "/camera/segmentation/road_mask",
            self.road_mask_callback,
            qos
        )

        # ==========================================================
        # PUBLISHERS
        # ==========================================================

        self.lane_overlay_pub = self.create_publisher(
            Image,
            "/camera/lane/overlay",
            qos
        )

        self.lane_mask_pub = self.create_publisher(
            Image,
            "/camera/lane/mask",
            qos
        )

        # ==========================================================
        # FRAME BUFFER
        # ==========================================================

        self.latest_image = None
        self.latest_road_mask = None
        self.latest_header = None

        self.processed_frames = 0

        self.get_logger().info(
            "Waiting for camera and road segmentation..."
        )

    # ==============================================================
    # CAMERA CALLBACK
    # ==============================================================

    def image_callback(self, msg):

        try:

            image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

        except Exception as e:

            self.get_logger().error(
                f"Camera conversion failed: {e}"
            )

            return

        self.latest_image = image
        self.latest_header = msg.header

        self.process_frame()

    # ==============================================================
    # ROAD MASK CALLBACK
    # ==============================================================

    def road_mask_callback(self, msg):

        try:

            road_mask = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="mono8"
            )

        except Exception as e:

            self.get_logger().error(
                f"Road mask conversion failed: {e}"
            )

            return

        self.latest_road_mask = road_mask

    # ==============================================================
    # MAIN PROCESSING
    # ==============================================================

    def process_frame(self):

        if self.latest_image is None:
            return

        if self.latest_road_mask is None:
            return

        image = self.latest_image

        road_mask = self.latest_road_mask

        # ----------------------------------------------------------
        # Make sure dimensions match
        # ----------------------------------------------------------

        if (
            image.shape[0] != road_mask.shape[0]
            or image.shape[1] != road_mask.shape[1]
        ):

            road_mask = cv2.resize(
                road_mask,
                (
                    image.shape[1],
                    image.shape[0]
                ),
                interpolation=cv2.INTER_NEAREST
            )

        # ==========================================================
        # STEP 1
        # ROAD REGION
        # ==========================================================

        road_binary = np.zeros_like(
            road_mask
        )

        road_binary[
            road_mask > 127
        ] = 255

        # ==========================================================
        # STEP 2
        # LOWER IMAGE ROI
        #
        # Lane markings are most useful in the lower portion
        # of the camera image.
        # ==========================================================

        h, w = image.shape[:2]

        roi_mask = np.zeros_like(
            road_binary
        )

        polygon = np.array(
            [
                [
                    int(0.05 * w),
                    h
                ],
                [
                    int(0.95 * w),
                    h
                ],
                [
                    int(0.65 * w),
                    int(0.48 * h)
                ],
                [
                    int(0.35 * w),
                    int(0.48 * h)
                ]
            ],
            dtype=np.int32
        )

        cv2.fillPoly(
            roi_mask,
            [polygon],
            255
        )

        road_roi = cv2.bitwise_and(
            road_binary,
            roi_mask
        )

        # ==========================================================
        # STEP 3
        # EDGE DETECTION
        # ==========================================================

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Smooth image before Canny.
        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        edges = cv2.Canny(
            blurred,
            50,
            150
        )

        # Keep only road pixels.
        lane_edges = cv2.bitwise_and(
            edges,
            road_roi
        )

        # ==========================================================
        # STEP 4
        # DETECT LINE SEGMENTS
        # ==========================================================

        lines = cv2.HoughLinesP(
            lane_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=35,
            minLineLength=30,
            maxLineGap=50
        )

        # ==========================================================
        # OUTPUT MASK
        # ==========================================================

        lane_mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        # ==========================================================
        # OUTPUT OVERLAY
        # ==========================================================

        overlay = image.copy()

        # Draw road region very lightly.
        road_color = np.zeros_like(
            image
        )

        road_color[
            road_binary > 127
        ] = (40, 80, 40)

        overlay = cv2.addWeighted(
            overlay,
            0.85,
            road_color,
            0.15,
            0
        )

        # ==========================================================
        # CLASSIFY LINE SEGMENTS
        # ==========================================================

        left_lines = []
        right_lines = []

        if lines is not None:

            for line in lines:

                x1, y1, x2, y2 = line[0]

                dx = x2 - x1
                dy = y2 - y1

                if abs(dx) < 5:
                    continue

                slope = dy / float(dx)

                # Ignore nearly horizontal lines.
                if abs(slope) < 0.3:
                    continue

                # --------------------------------------------------
                # LEFT / RIGHT CLASSIFICATION
                # --------------------------------------------------

                x_mid = (x1 + x2) / 2.0

                image_center = w / 2.0

                if slope < 0 and x_mid < image_center:

                    left_lines.append(
                        (x1, y1, x2, y2)
                    )

                elif slope > 0 and x_mid > image_center:

                    right_lines.append(
                        (x1, y1, x2, y2)
                    )

        # ==========================================================
        # FIT LANE LINES
        # ==========================================================

        left_lane = self.fit_lane(
            left_lines,
            h
        )

        right_lane = self.fit_lane(
            right_lines,
            h
        )

        # ==========================================================
        # DRAW LEFT LANE
        # ==========================================================

        if left_lane is not None:

            x1, y1, x2, y2 = left_lane

            cv2.line(
                overlay,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                5
            )

            cv2.line(
                lane_mask,
                (x1, y1),
                (x2, y2),
                255,
                5
            )

        # ==========================================================
        # DRAW RIGHT LANE
        # ==========================================================

        if right_lane is not None:

            x1, y1, x2, y2 = right_lane

            cv2.line(
                overlay,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                5
            )

            cv2.line(
                lane_mask,
                (x1, y1),
                (x2, y2),
                255,
                5
            )

        # ==========================================================
        # DRAW LANE CORRIDOR
        # ==========================================================

        if (
            left_lane is not None
            and right_lane is not None
        ):

            lx1, ly1, lx2, ly2 = left_lane
            rx1, ry1, rx2, ry2 = right_lane

            polygon = np.array(
                [
                    [lx1, ly1],
                    [rx1, ry1],
                    [rx2, ry2],
                    [lx2, ly2]
                ],
                dtype=np.int32
            )

            corridor = overlay.copy()

            cv2.fillPoly(
                corridor,
                [polygon],
                (0, 100, 100)
            )

            overlay = cv2.addWeighted(
                overlay,
                0.75,
                corridor,
                0.25,
                0
            )

        # ==========================================================
        # TEXT
        # ==========================================================

        cv2.putText(
            overlay,
            "Lane Detection",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )

        # ==========================================================
        # PUBLISH
        # ==========================================================

        overlay_msg = self.bridge.cv2_to_imgmsg(
            overlay,
            encoding="bgr8"
        )

        overlay_msg.header = self.latest_header

        self.lane_overlay_pub.publish(
            overlay_msg
        )

        mask_msg = self.bridge.cv2_to_imgmsg(
            lane_mask,
            encoding="mono8"
        )

        mask_msg.header = self.latest_header

        self.lane_mask_pub.publish(
            mask_msg
        )

        self.processed_frames += 1

        if self.processed_frames % 20 == 0:

            self.get_logger().info(
                "Lane detection | "
                f"processed={self.processed_frames} | "
                f"left={left_lane is not None} | "
                f"right={right_lane is not None}"
            )

    # ==============================================================
    # FIT LANE
    # ==============================================================

    def fit_lane(
        self,
        lines,
        image_height
    ):

        if len(lines) < 2:
            return None

        points_x = []
        points_y = []

        for x1, y1, x2, y2 in lines:

            points_x.extend(
                [x1, x2]
            )

            points_y.extend(
                [y1, y2]
            )

        if len(points_x) < 4:
            return None

        try:

            # Fit x = a*y + b.
            #
            # Using y as the independent variable makes the
            # representation convenient for approximately vertical
            # lane boundaries.

            coeff = np.polyfit(
                points_y,
                points_x,
                1
            )

            a, b = coeff

        except Exception:

            return None

        y_bottom = image_height

        y_top = int(
            0.50 * image_height
        )

        x_bottom = int(
            a * y_bottom + b
        )

        x_top = int(
            a * y_top + b
        )

        # Keep inside image.
        x_bottom = max(
            0,
            min(image_height * 2, x_bottom)
        )

        x_top = max(
            0,
            min(image_height * 2, x_top)
        )

        return (
            int(x_bottom),
            int(y_bottom),
            int(x_top),
            int(y_top)
        )


# ==================================================================
# MAIN
# ==================================================================

def main(args=None):

    rclpy.init(args=args)

    node = LaneDetectionNode()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()
