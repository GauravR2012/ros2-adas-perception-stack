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

        # ==========================================================
        # TEMPORAL TRACKING PARAMETERS
        # ==========================================================

        # EMA smoothing factor.
        #
        # Smaller alpha:
        #   more smoothing
        #   slower response
        #
        # Larger alpha:
        #   less smoothing
        #   faster response
        #
        self.TRACK_ALPHA = 0.40 #0.15 #0.40 #0.25

        # Number of consecutive frames for which a missing
        # measurement can be tolerated before the lane state
        # is discarded.
        self.MAX_MISSED_FRAMES = 8

        # ==========================================================
        # TEMPORAL LANE STATE
        # ==========================================================

        # Each lane is represented by:
        #
        # [x_bottom, y_bottom, x_top, y_top]
        #
        self.tracked_left_lane = None
        self.tracked_right_lane = None

        # Number of consecutive frames in which a lane
        # measurement was unavailable.
        self.left_missed_frames = 0
        self.right_missed_frames = 0

        # ==========================================================
        # STATISTICS
        # ==========================================================

        self.processed_frames = 0

        self.get_logger().info(
            "Temporal lane tracking enabled."
        )

        self.get_logger().info(
            f"EMA alpha: {self.TRACK_ALPHA}"
        )

        self.get_logger().info(
            f"Max missed frames: {self.MAX_MISSED_FRAMES}"
        )

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

        image = self.latest_image.copy()
        road_mask = self.latest_road_mask.copy()

        h, w = image.shape[:2]

        # ==========================================================
        # 1. RESIZE ROAD MASK
        # ==========================================================

        if (
            road_mask.shape[0] != h
            or road_mask.shape[1] != w
        ):

            road_mask = cv2.resize(
                road_mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )

        # ==========================================================
        # 2. BINARY ROAD MASK
        # ==========================================================

        road_binary = np.zeros_like(
            road_mask
        )

        road_binary[
            road_mask > 127
        ] = 255

        # ==========================================================
        # 3. LOWER CAMERA ROI
        # ==========================================================

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
                    int(0.68 * w),
                    int(0.48 * h)
                ],
                [
                    int(0.32 * w),
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

        # ==========================================================
        # 4. ROAD + ROI
        # ==========================================================

        road_roi = cv2.bitwise_and(
            road_binary,
            roi_mask
        )

        # ==========================================================
        # 5. GRAYSCALE
        # ==========================================================

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # ==========================================================
        # 6. GAUSSIAN BLUR
        # ==========================================================

        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        # ==========================================================
        # 7. CANNY
        # ==========================================================

        edges = cv2.Canny(
            blurred,
            50,
            150
        )

        # ==========================================================
        # 8. CONSTRAIN EDGES TO ROAD
        # ==========================================================

        road_edges = cv2.bitwise_and(
            edges,
            road_roi
        )

        # ==========================================================
        # 9. MORPHOLOGICAL CLEANING
        # ==========================================================

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        road_edges = cv2.morphologyEx(
            road_edges,
            cv2.MORPH_CLOSE,
            kernel
        )

        # ==========================================================
        # 10. HOUGH LINE DETECTION
        # ==========================================================

        lines = cv2.HoughLinesP(
            road_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=50,
            maxLineGap=60
        )

        # ==========================================================
        # 11. CLASSIFY LINE SEGMENTS
        # ==========================================================

        left_lines = []
        right_lines = []

        image_center = w / 2.0

        if lines is not None:

            for line in lines:

                x1, y1, x2, y2 = line[0]

                dx = x2 - x1
                dy = y2 - y1

                if abs(dx) < 5:
                    continue

                slope = dy / float(dx)

                # --------------------------------------------------
                # Keep current filtering unchanged.
                # We are deliberately NOT making it more restrictive.
                # --------------------------------------------------

                if abs(slope) < 0.35:
                    continue

                if abs(slope) > 5.0:
                    continue

                x_mid = (
                    x1 + x2
                ) / 2.0

                # --------------------------------------------------
                # LEFT
                # --------------------------------------------------

                if (
                    slope < 0
                    and x_mid < image_center
                ):

                    left_lines.append(
                        (
                            x1,
                            y1,
                            x2,
                            y2
                        )
                    )

                # --------------------------------------------------
                # RIGHT
                # --------------------------------------------------

                elif (
                    slope > 0
                    and x_mid > image_center
                ):

                    right_lines.append(
                        (
                            x1,
                            y1,
                            x2,
                            y2
                        )
                    )

        # ==========================================================
        # 12. CURRENT FRAME LANE MEASUREMENTS
        # ==========================================================

        left_measurement = self.fit_lane(
            left_lines,
            h
        )

        right_measurement = self.fit_lane(
            right_lines,
            h
        )

        # ==========================================================
        # 13. TEMPORAL TRACKING
        # ==========================================================

        left_lane = self.update_lane_track(
            left_measurement,
            is_left=True
        )

        right_lane = self.update_lane_track(
            right_measurement,
            is_left=False
        )

        # ==========================================================
        # 14. LANE MASK
        # ==========================================================

        lane_mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        # ==========================================================
        # 15. VISUALIZATION
        # ==========================================================

        overlay = image.copy()

        # ----------------------------------------------------------
        # Show road segmentation subtly.
        # ----------------------------------------------------------

        road_color = np.zeros_like(
            image
        )

        road_color[
            road_binary > 127
        ] = (
            40,
            80,
            40
        )

        overlay = cv2.addWeighted(
            overlay,
            0.90,
            road_color,
            0.10,
            0
        )

        # ==========================================================
        # 16. LEFT LANE
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
        # 17. RIGHT LANE
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
        # 18. LANE CORRIDOR
        # ==========================================================

        if (
            left_lane is not None
            and right_lane is not None
        ):

            lx1, ly1, lx2, ly2 = left_lane
            rx1, ry1, rx2, ry2 = right_lane

            corridor = np.array(
                [
                    [lx1, ly1],
                    [rx1, ry1],
                    [rx2, ry2],
                    [lx2, ly2]
                ],
                dtype=np.int32
            )

            corridor_overlay = overlay.copy()

            cv2.fillPoly(
                corridor_overlay,
                [corridor],
                (0, 180, 180)
            )

            overlay = cv2.addWeighted(
                overlay,
                0.75,
                corridor_overlay,
                0.25,
                0
            )

        # ==========================================================
        # 19. DEBUG INFORMATION
        # ==========================================================

        cv2.putText(
            overlay,
            "Lane Detection + Temporal Tracking",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )

        cv2.putText(
            overlay,
            (
                f"Left: "
                f"{'TRACKED' if left_lane is not None else '---'}"
            ),
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            overlay,
            (
                f"Right: "
                f"{'TRACKED' if right_lane is not None else '---'}"
            ),
            (30, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            overlay,
            (
                f"Missed L/R: "
                f"{self.left_missed_frames}/"
                f"{self.right_missed_frames}"
            ),
            (30, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # ==========================================================
        # 20. PUBLISH OVERLAY
        # ==========================================================

        overlay_msg = self.bridge.cv2_to_imgmsg(
            overlay,
            encoding="bgr8"
        )

        overlay_msg.header = self.latest_header

        self.lane_overlay_pub.publish(
            overlay_msg
        )

        # ==========================================================
        # 21. PUBLISH LANE MASK
        # ==========================================================

        mask_msg = self.bridge.cv2_to_imgmsg(
            lane_mask,
            encoding="mono8"
        )

        mask_msg.header = self.latest_header

        self.lane_mask_pub.publish(
            mask_msg
        )

        # ==========================================================
        # 22. LOGGING
        # ==========================================================

        self.processed_frames += 1

        if self.processed_frames % 20 == 0:

            self.get_logger().info(
                "Lane detection | "
                f"processed={self.processed_frames} | "
                f"raw_left={left_measurement is not None} | "
                f"raw_right={right_measurement is not None} | "
                f"tracked_left={left_lane is not None} | "
                f"tracked_right={right_lane is not None} | "
                f"segments={0 if lines is None else len(lines)} | "
                f"missed={self.left_missed_frames}/"
                f"{self.right_missed_frames}"
            )

    # ==============================================================
    # TEMPORAL LANE TRACKER
    # ==============================================================

    def update_lane_track(
        self,
        measurement,
        is_left
    ):
        """
        Update a lane using exponential moving average.

        measurement:
            [x_bottom, y_bottom, x_top, y_top]

        The lane is maintained for a limited number of frames
        when the current Hough detector fails to produce a
        measurement.
        """

        if is_left:

            previous = self.tracked_left_lane

            # ------------------------------------------------------
            # Valid measurement
            # ------------------------------------------------------

            if measurement is not None:

                if previous is None:

                    # First measurement initializes the track.
                    tracked = np.array(
                        measurement,
                        dtype=np.float32
                    )

                else:

                    current = np.array(
                        measurement,
                        dtype=np.float32
                    )

                    previous_array = np.array(
                        previous,
                        dtype=np.float32
                    )

                    # EMA update.
                    tracked = (
                        self.TRACK_ALPHA * current
                        +
                        (1.0 - self.TRACK_ALPHA)
                        * previous_array
                    )

                self.tracked_left_lane = (
                    tracked.astype(np.int32)
                )

                self.left_missed_frames = 0

                return tuple(
                    self.tracked_left_lane.tolist()
                )

            # ------------------------------------------------------
            # Missing measurement
            # ------------------------------------------------------

            self.left_missed_frames += 1

            if (
                previous is not None
                and self.left_missed_frames
                <= self.MAX_MISSED_FRAMES
            ):

                # Keep previous estimate temporarily.
                return tuple(
                    np.asarray(
                        previous,
                        dtype=np.int32
                    ).tolist()
                )

            # Track has been missing for too long.
            self.tracked_left_lane = None

            return None

        # ==========================================================
        # RIGHT LANE
        # ==========================================================

        previous = self.tracked_right_lane

        # ----------------------------------------------------------
        # Valid measurement
        # ----------------------------------------------------------

        if measurement is not None:

            if previous is None:

                tracked = np.array(
                    measurement,
                    dtype=np.float32
                )

            else:

                current = np.array(
                    measurement,
                    dtype=np.float32
                )

                previous_array = np.array(
                    previous,
                    dtype=np.float32
                )

                # EMA update.
                tracked = (
                    self.TRACK_ALPHA * current
                    +
                    (1.0 - self.TRACK_ALPHA)
                    * previous_array
                )

            self.tracked_right_lane = (
                tracked.astype(np.int32)
            )

            self.right_missed_frames = 0

            return tuple(
                self.tracked_right_lane.tolist()
            )

        # ----------------------------------------------------------
        # Missing measurement
        # ----------------------------------------------------------

        self.right_missed_frames += 1

        if (
            previous is not None
            and self.right_missed_frames
            <= self.MAX_MISSED_FRAMES
        ):

            return tuple(
                np.asarray(
                    previous,
                    dtype=np.int32
                ).tolist()
            )

        # Track has been missing for too long.
        self.tracked_right_lane = None

        return None

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

        # ==========================================================
        # COLLECT LINE POINTS
        # ==========================================================

        for x1, y1, x2, y2 in lines:

            points_x.extend(
                [x1, x2]
            )

            points_y.extend(
                [y1, y2]
            )

        if len(points_x) < 4:
            return None

        # ==========================================================
        # FIT
        #
        # x = a*y + b
        #
        # We keep the existing linear model.
        # Polynomial fitting comes later.
        # ==========================================================

        try:

            coeff = np.polyfit(
                points_y,
                points_x,
                1
            )

            a, b = coeff

        except Exception:

            return None

        # ==========================================================
        # LANE ENDPOINTS
        # ==========================================================

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

        # ==========================================================
        # SANITY CHECK
        # ==========================================================

        if (
            x_bottom < -image_height
            or x_bottom > 2 * image_height
        ):
            return None

        if (
            x_top < -image_height
            or x_top > 2 * image_height
        ):
            return None

        return (
            x_bottom,
            y_bottom,
            x_top,
            y_top
        )


# ==================================================================
# MAIN
# ==================================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = LaneDetectionNode()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()