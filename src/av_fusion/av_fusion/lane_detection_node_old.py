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
        # TEMPORAL TRACKING
        # ==========================================================

        self.TRACK_ALPHA = 0.40

        self.MAX_MISSED_FRAMES = 8

        # ==========================================================
        # LANE STATE
        #
        # Straight-line representation:
        #
        #       x = a*y + b
        #
        # State = [a, b]
        # ==========================================================

        self.tracked_left_lane = None
        self.tracked_right_lane = None

        self.left_missed_frames = 0
        self.right_missed_frames = 0

        # ==========================================================
        # STATISTICS
        # ==========================================================

        self.processed_frames = 0

        self.get_logger().info(
            "Straight-line lane fitting enabled."
        )

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
        # 3. ROI
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
        # 6. BLUR
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
        # 8. ROAD-CONSTRAINED EDGES
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
        # 10. HOUGH
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
        # 11. LEFT / RIGHT CLASSIFICATION
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

                # Keep the same filtering as the working version.

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
        # 12. FIT CURRENT LANE MEASUREMENTS
        # ==========================================================

        left_measurement = self.fit_lane(
            left_lines
        )

        right_measurement = self.fit_lane(
            right_lines
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
        # 14. CREATE OUTPUTS
        # ==========================================================

        overlay = image.copy()

        lane_mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        # ==========================================================
        # 15. SUBTLE ROAD VISUALIZATION
        # ==========================================================

        road_visualization = np.zeros_like(
            image
        )

        road_visualization[
            road_binary > 127
        ] = (
            40,
            80,
            40
        )

        overlay = cv2.addWeighted(
            overlay,
            0.90,
            road_visualization,
            0.10,
            0
        )

        # ==========================================================
        # 16. SAMPLE Y VALUES
        # ==========================================================

        y_bottom = h - 1
        y_top = int(0.50 * h)

        y_values = np.linspace(
            y_top,
            y_bottom,
            100
        ).astype(np.int32)

        # ==========================================================
        # 17. EVALUATE LANES
        # ==========================================================

        left_points = None
        right_points = None

        if left_lane is not None:

            left_points = self.evaluate_lane(
                left_lane,
                y_values,
                w
            )

            if len(left_points) >= 2:

                cv2.polylines(
                    overlay,
                    [left_points],
                    False,
                    (0, 255, 255),
                    5
                )

                cv2.polylines(
                    lane_mask,
                    [left_points],
                    False,
                    255,
                    5
                )

        if right_lane is not None:

            right_points = self.evaluate_lane(
                right_lane,
                y_values,
                w
            )

            if len(right_points) >= 2:

                cv2.polylines(
                    overlay,
                    [right_points],
                    False,
                    (0, 255, 255),
                    5
                )

                cv2.polylines(
                    lane_mask,
                    [right_points],
                    False,
                    255,
                    5
                )

        # ==========================================================
        # 18. LANE CENTERLINE
        # ==========================================================

        center_points = None
        lane_width_bottom = None
        center_offset = None

        if (
            left_lane is not None
            and right_lane is not None
        ):

            center_points = self.compute_centerline(
                left_lane,
                right_lane,
                y_values,
                w
            )

            if len(center_points) >= 2:

                # Green = lane center.
                cv2.polylines(
                    overlay,
                    [center_points],
                    False,
                    (0, 255, 0),
                    5
                )

                cv2.polylines(
                    lane_mask,
                    [center_points],
                    False,
                    128,
                    5
                )

            # ======================================================
            # LANE WIDTH AT VEHICLE REGION
            # ======================================================

            y_eval = y_bottom

            x_left = self.evaluate_x(
                left_lane,
                y_eval
            )

            x_right = self.evaluate_x(
                right_lane,
                y_eval
            )

            x_center = (
                x_left + x_right
            ) / 2.0

            lane_width_bottom = (
                x_right - x_left
            )

            image_center = w / 2.0

            center_offset = (
                x_center - image_center
            )

            # ======================================================
            # DRAW CENTER POINT
            # ======================================================

            if (
                0 <= x_center < w
            ):

                cv2.circle(
                    overlay,
                    (
                        int(x_center),
                        int(y_eval)
                    ),
                    10,
                    (0, 255, 0),
                    -1
                )

                # Image center reference.
                cv2.line(
                    overlay,
                    (
                        int(image_center),
                        h
                    ),
                    (
                        int(image_center),
                        int(0.80 * h)
                    ),
                    (255, 0, 0),
                    3
                )

        # ==========================================================
        # 19. LANE CORRIDOR
        # ==========================================================

        if (
            left_points is not None
            and right_points is not None
            and len(left_points) >= 2
            and len(right_points) >= 2
        ):

            corridor_points = np.vstack(
                [
                    left_points,
                    right_points[::-1]
                ]
            )

            corridor_overlay = overlay.copy()

            cv2.fillPoly(
                corridor_overlay,
                [corridor_points],
                (0, 180, 180)
            )

            overlay = cv2.addWeighted(
                overlay,
                0.78,
                corridor_overlay,
                0.22,
                0
            )

            # Redraw lanes after corridor.

            cv2.polylines(
                overlay,
                [left_points],
                False,
                (0, 255, 255),
                5
            )

            cv2.polylines(
                overlay,
                [right_points],
                False,
                (0, 255, 255),
                5
            )

            if center_points is not None:

                cv2.polylines(
                    overlay,
                    [center_points],
                    False,
                    (0, 255, 0),
                    5
                )

        # ==========================================================
        # 20. DEBUG TEXT
        # ==========================================================

        cv2.putText(
            overlay,
            "Lane Detection + Temporal Tracking",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            overlay,
            (
                "Left: "
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
                "Right: "
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
                "Center: "
                f"{'AVAILABLE' if center_points is not None else '---'}"
            ),
            (30, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            overlay,
            (
                f"Missed L/R: "
                f"{self.left_missed_frames}/"
                f"{self.right_missed_frames}"
            ),
            (30, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # ==========================================================
        # LANE WIDTH
        # ==========================================================

        if lane_width_bottom is not None:

            cv2.putText(
                overlay,
                (
                    f"Lane width: "
                    f"{lane_width_bottom:.0f} px"
                ),
                (30, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # ==========================================================
        # CENTER OFFSET
        # ==========================================================

        if center_offset is not None:

            cv2.putText(
                overlay,
                (
                    f"Center offset: "
                    f"{center_offset:+.0f} px"
                ),
                (30, 225),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # ==========================================================
        # 21. PUBLISH OVERLAY
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
        # 22. PUBLISH LANE MASK
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
        # 23. LOGGING
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
                f"center={center_points is not None} | "
                f"segments="
                f"{0 if lines is None else len(lines)} | "
                f"missed="
                f"{self.left_missed_frames}/"
                f"{self.right_missed_frames}"
            )

    # ==============================================================
    # FIT STRAIGHT LANE
    # ==============================================================

    def fit_lane(
        self,
        lines
    ):
        """
        Fit:

            x = a*y + b

        to Hough line endpoints.

        Returns:

            [a, b]
        """

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

        points_x = np.asarray(
            points_x,
            dtype=np.float64
        )

        points_y = np.asarray(
            points_y,
            dtype=np.float64
        )

        try:

            coeffs = np.polyfit(
                points_y,
                points_x,
                1
            )

        except Exception:

            return None

        if not np.all(
            np.isfinite(coeffs)
        ):
            return None

        return coeffs

    # ==============================================================
    # TEMPORAL TRACKING
    # ==============================================================

    def update_lane_track(
        self,
        measurement,
        is_left
    ):

        alpha = self.TRACK_ALPHA

        # ==========================================================
        # LEFT
        # ==========================================================

        if is_left:

            previous = self.tracked_left_lane

            if measurement is not None:

                current = np.asarray(
                    measurement,
                    dtype=np.float64
                )

                if previous is None:

                    tracked = current

                else:

                    previous = np.asarray(
                        previous,
                        dtype=np.float64
                    )

                    tracked = (
                        alpha * current
                        +
                        (1.0 - alpha) * previous
                    )

                self.tracked_left_lane = tracked

                self.left_missed_frames = 0

                return tracked

            self.left_missed_frames += 1

            if (
                previous is not None
                and self.left_missed_frames
                <= self.MAX_MISSED_FRAMES
            ):

                return previous

            self.tracked_left_lane = None

            return None

        # ==========================================================
        # RIGHT
        # ==========================================================

        previous = self.tracked_right_lane

        if measurement is not None:

            current = np.asarray(
                measurement,
                dtype=np.float64
            )

            if previous is None:

                tracked = current

            else:

                previous = np.asarray(
                    previous,
                    dtype=np.float64
                )

                tracked = (
                    alpha * current
                    +
                    (1.0 - alpha) * previous
                )

            self.tracked_right_lane = tracked

            self.right_missed_frames = 0

            return tracked

        self.right_missed_frames += 1

        if (
            previous is not None
            and self.right_missed_frames
            <= self.MAX_MISSED_FRAMES
        ):

            return previous

        self.tracked_right_lane = None

        return None

    # ==============================================================
    # EVALUATE LANE
    # ==============================================================

    def evaluate_lane(
        self,
        coeffs,
        y_values,
        image_width
    ):

        a, b = coeffs

        y_values = np.asarray(
            y_values,
            dtype=np.float64
        )

        x_values = (
            a * y_values
            + b
        )

        valid = (
            np.isfinite(x_values)
            &
            (x_values >= 0)
            &
            (x_values < image_width)
        )

        if not np.any(valid):

            return np.empty(
                (0, 1, 2),
                dtype=np.int32
            )

        x_values = x_values[valid]
        y_values = y_values[valid]

        points = np.column_stack(
            [
                x_values,
                y_values
            ]
        ).astype(
            np.int32
        )

        return points.reshape(
            (-1, 1, 2)
        )

    # ==============================================================
    # EVALUATE X
    # ==============================================================

    def evaluate_x(
        self,
        coeffs,
        y
    ):

        a, b = coeffs

        return (
            a * y
            + b
        )

    # ==============================================================
    # COMPUTE CENTERLINE
    # ==============================================================

    def compute_centerline(
        self,
        left_lane,
        right_lane,
        y_values,
        image_width
    ):

        left_a, left_b = left_lane
        right_a, right_b = right_lane

        y_values = np.asarray(
            y_values,
            dtype=np.float64
        )

        x_left = (
            left_a * y_values
            + left_b
        )

        x_right = (
            right_a * y_values
            + right_b
        )

        x_center = (
            x_left + x_right
        ) / 2.0

        valid = (
            np.isfinite(x_center)
            &
            (x_center >= 0)
            &
            (x_center < image_width)
        )

        if not np.any(valid):

            return np.empty(
                (0, 1, 2),
                dtype=np.int32
            )

        points = np.column_stack(
            [
                x_center[valid],
                y_values[valid]
            ]
        ).astype(
            np.int32
        )

        return points.reshape(
            (-1, 1, 2)
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