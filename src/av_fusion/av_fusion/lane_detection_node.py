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
        # IMAGE BUFFER
        # ==========================================================

        self.latest_image = None
        self.latest_road_mask = None
        self.latest_header = None

        # ==========================================================
        # TEMPORAL LANE TRACKING
        # ==========================================================

        # Higher alpha = faster response.
        # 0.4 worked reasonably well in previous experiments.
        self.TRACK_ALPHA = 0.40

        # Keep lane alive when detector temporarily misses it.
        self.MAX_MISSED_FRAMES = 8

        # ==========================================================
        # SOFT GEOMETRIC VALIDATION
        # ==========================================================

        # Expected lane width relative to image width.
        self.MIN_LANE_WIDTH_RATIO = 0.20
        self.MAX_LANE_WIDTH_RATIO = 0.95

        # Do not allow the center to be completely unreasonable.
        self.MAX_CENTER_OFFSET_RATIO = 0.50

        # ==========================================================
        # TEMPORAL MEASUREMENT GATES
        # ==========================================================

        # Previous hard gate was too restrictive.
        # These are intentionally generous.
        self.MAX_BOTTOM_X_JUMP_RATIO = 0.25
        self.MAX_SLOPE_CHANGE = 0.65

        # ==========================================================
        # GEOMETRY SMOOTHING
        # ==========================================================

        self.GEOMETRY_ALPHA = 0.20

        self.smoothed_width = None
        self.smoothed_offset = None
        self.smoothed_heading = None

        # ==========================================================
        # HOUGH PARAMETERS
        # ==========================================================

        self.HOUGH_THRESHOLD = 30

        self.MIN_LINE_LENGTH = 40

        self.MAX_LINE_GAP = 80

        # ==========================================================
        # ROI
        # ==========================================================

        self.ROI_TOP_RATIO = 0.48

        # ==========================================================
        # STATE
        # ==========================================================

        self.tracked_left_lane = None
        self.tracked_right_lane = None

        self.left_missed_frames = 0
        self.right_missed_frames = 0

        self.processed_frames = 0

        # Diagnostics.
        self.left_rejected = 0
        self.right_rejected = 0
        self.geometry_warnings = 0

        # ==========================================================
        # LOGGING
        # ==========================================================

        self.get_logger().info(
            "Straight-line lane fitting enabled."
        )

        self.get_logger().info(
            "Temporal lane tracking enabled."
        )

        self.get_logger().info(
            f"Lane EMA alpha: {self.TRACK_ALPHA}"
        )

        self.get_logger().info(
            f"Geometry EMA alpha: {self.GEOMETRY_ALPHA}"
        )

        self.get_logger().info(
            f"Max missed frames: {self.MAX_MISSED_FRAMES}"
        )

        self.get_logger().info(
            "Soft lane-pair validation enabled."
        )

        self.get_logger().info(
            "Temporal geometry smoothing enabled."
        )

        self.get_logger().info(
            "Waiting for camera and road segmentation..."
        )

    # ==============================================================
    # CAMERA CALLBACK
    # ==============================================================

    def image_callback(self, msg):

        try:

            self.latest_image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

        except Exception as e:

            self.get_logger().error(
                f"Camera conversion failed: {e}"
            )

            return

        self.latest_header = msg.header

        self.process_frame()

    # ==============================================================
    # ROAD MASK CALLBACK
    # ==============================================================

    def road_mask_callback(self, msg):

        try:

            self.latest_road_mask = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="mono8"
            )

        except Exception as e:

            self.get_logger().error(
                f"Road mask conversion failed: {e}"
            )

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
        # RESIZE ROAD MASK
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
        # BINARY ROAD MASK
        # ==========================================================

        road_binary = np.zeros_like(
            road_mask
        )

        road_binary[
            road_mask > 127
        ] = 255

        # ==========================================================
        # ROI
        # ==========================================================

        roi_mask = np.zeros_like(
            road_binary
        )

        polygon = np.array(
            [
                [
                    int(0.05 * w),
                    h - 1
                ],
                [
                    int(0.95 * w),
                    h - 1
                ],
                [
                    int(0.70 * w),
                    int(self.ROI_TOP_RATIO * h)
                ],
                [
                    int(0.30 * w),
                    int(self.ROI_TOP_RATIO * h)
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
        # IMAGE EDGES
        # ==========================================================

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

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

        # Only edges inside road.
        road_edges = cv2.bitwise_and(
            edges,
            road_roi
        )

        # ==========================================================
        # MORPHOLOGICAL CLEANUP
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
        # HOUGH
        # ==========================================================

        lines = cv2.HoughLinesP(
            road_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.HOUGH_THRESHOLD,
            minLineLength=self.MIN_LINE_LENGTH,
            maxLineGap=self.MAX_LINE_GAP
        )

        # ==========================================================
        # CLASSIFY LINE SEGMENTS
        # ==========================================================

        left_lines = []
        right_lines = []

        image_center = w / 2.0

        if lines is not None:

            for line in lines:

                x1, y1, x2, y2 = line[0]

                dx = float(x2 - x1)
                dy = float(y2 - y1)

                if abs(dx) < 5:
                    continue

                slope = dy / dx

                # Ignore almost-horizontal structures.
                if abs(slope) < 0.30:
                    continue

                # Ignore nearly vertical noise.
                if abs(slope) > 6.0:
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
        # FIT RAW LANES
        # ==========================================================

        raw_left = self.fit_lane(
            left_lines
        )

        raw_right = self.fit_lane(
            right_lines
        )

        # ==========================================================
        # SOFT TEMPORAL GATING
        # ==========================================================

        left_measurement = self.temporal_gate(
            raw_left,
            self.tracked_left_lane,
            h,
            w,
            True
        )

        right_measurement = self.temporal_gate(
            raw_right,
            self.tracked_right_lane,
            h,
            w,
            False
        )

        # ==========================================================
        # UPDATE TRACKS
        # ==========================================================

        left_lane = self.update_lane_track(
            left_measurement,
            True
        )

        right_lane = self.update_lane_track(
            right_measurement,
            False
        )

        # ==========================================================
        # GEOMETRY
        # ==========================================================

        geometry = None

        if (
            left_lane is not None
            and right_lane is not None
        ):

            geometry = self.compute_lane_geometry(
                left_lane,
                right_lane,
                h,
                w
            )

        # ==========================================================
        # OVERLAY
        # ==========================================================

        overlay = image.copy()

        # ==========================================================
        # ROAD VISUALIZATION
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
        # LANE MASK
        # ==========================================================

        lane_mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        # ==========================================================
        # DRAW LANES
        # ==========================================================

        y_values = np.linspace(
            int(0.50 * h),
            h - 1,
            100
        ).astype(np.int32)

        left_points = None
        right_points = None

        # ----------------------------------------------------------
        # LEFT
        # ----------------------------------------------------------

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

        # ----------------------------------------------------------
        # RIGHT
        # ----------------------------------------------------------

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
        # LANE CORRIDOR + CENTERLINE
        # ==========================================================

        center_points = None

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
                0.82,
                corridor_overlay,
                0.18,
                0
            )

            # Redraw boundaries.
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

            # ------------------------------------------------------
            # Centerline
            # ------------------------------------------------------

            center_points = self.compute_center_points(
                left_lane,
                right_lane,
                y_values,
                w
            )

            if center_points is not None:

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

        # ==========================================================
        # IMAGE CENTER
        # ==========================================================

        cv2.line(
            overlay,
            (
                int(w / 2),
                h
            ),
            (
                int(w / 2),
                int(0.70 * h)
            ),
            (255, 0, 0),
            3
        )

        # ==========================================================
        # LOOK-AHEAD POINTS
        # ==========================================================

        if geometry is not None:

            for point in geometry[
                "lookahead_points"
            ]:

                x = int(
                    point["x"]
                )

                y = int(
                    point["y"]
                )

                if (
                    0 <= x < w
                    and 0 <= y < h
                ):

                    cv2.circle(
                        overlay,
                        (x, y),
                        6,
                        (255, 0, 255),
                        -1
                    )

        # ==========================================================
        # DEBUG TEXT
        # ==========================================================

        cv2.putText(
            overlay,
            "Lane Detection - Temporal Geometry",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            (0, 255, 255),
            2
        )

        cv2.putText(
            overlay,
            (
                "Left: "
                +
                (
                    "TRACKED"
                    if left_lane is not None
                    else "---"
                )
            ),
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            overlay,
            (
                "Right: "
                +
                (
                    "TRACKED"
                    if right_lane is not None
                    else "---"
                )
            ),
            (30, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        # ==========================================================
        # GEOMETRY INFORMATION
        # ==========================================================

        if geometry is not None:

            cv2.putText(
                overlay,
                (
                    f"Width: "
                    f"{geometry['width']:.0f}px"
                ),
                (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.putText(
                overlay,
                (
                    f"Offset: "
                    f"{geometry['offset']:+.0f}px"
                ),
                (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                overlay,
                (
                    f"Heading: "
                    f"{geometry['heading']:+.1f} deg"
                ),
                (30, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                overlay,
                (
                    f"Width std: "
                    f"{geometry['width_std']:.1f}px"
                ),
                (30, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 0, 255),
                2
            )

            cv2.putText(
                overlay,
                "Geometry: VALID",
                (30, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                overlay,
                "Geometry: WAITING",
                (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2
            )

        # ==========================================================
        # TRACKING INFO
        # ==========================================================

        cv2.putText(
            overlay,
            (
                f"Missed L/R: "
                f"{self.left_missed_frames}/"
                f"{self.right_missed_frames}"
            ),
            (30, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2
        )

        cv2.putText(
            overlay,
            (
                f"Rejected L/R: "
                f"{self.left_rejected}/"
                f"{self.right_rejected}"
            ),
            (30, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 180, 0),
            2
        )

        # ==========================================================
        # PUBLISH OVERLAY
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
        # PUBLISH LANE MASK
        # ==========================================================

        lane_mask_msg = self.bridge.cv2_to_imgmsg(
            lane_mask,
            encoding="mono8"
        )

        lane_mask_msg.header = self.latest_header

        self.lane_mask_pub.publish(
            lane_mask_msg
        )

        # ==========================================================
        # LOGGING
        # ==========================================================

        self.processed_frames += 1

        if self.processed_frames % 20 == 0:

            if geometry is not None:

                self.get_logger().info(
                    "Lane geometry | "
                    f"processed={self.processed_frames} | "
                    f"raw_left={raw_left is not None} | "
                    f"raw_right={raw_right is not None} | "
                    f"tracked_left={left_lane is not None} | "
                    f"tracked_right={right_lane is not None} | "
                    f"width={geometry['width']:.0f}px | "
                    f"offset={geometry['offset']:+.0f}px | "
                    f"heading={geometry['heading']:+.1f}deg | "
                    f"width_std={geometry['width_std']:.1f}px | "
                    f"rejected="
                    f"{self.left_rejected}/"
                    f"{self.right_rejected}"
                )

            else:

                self.get_logger().info(
                    "Lane geometry | "
                    f"processed={self.processed_frames} | "
                    f"raw_left={raw_left is not None} | "
                    f"raw_right={raw_right is not None} | "
                    f"tracked_left={left_lane is not None} | "
                    f"tracked_right={right_lane is not None} | "
                    "geometry=UNAVAILABLE"
                )

    # ==============================================================
    # FIT LANE
    # ==============================================================

    def fit_lane(self, lines):

        if len(lines) < 2:
            return None

        xs = []
        ys = []

        weights = []

        for x1, y1, x2, y2 in lines:

            length = np.sqrt(
                (x2 - x1) ** 2
                +
                (y2 - y1) ** 2
            )

            xs.extend(
                [x1, x2]
            )

            ys.extend(
                [y1, y2]
            )

            weights.extend(
                [length, length]
            )

        if len(xs) < 4:
            return None

        xs = np.asarray(
            xs,
            dtype=np.float64
        )

        ys = np.asarray(
            ys,
            dtype=np.float64
        )

        weights = np.asarray(
            weights,
            dtype=np.float64
        )

        try:

            coeffs = np.polyfit(
                ys,
                xs,
                1,
                w=weights
            )

        except Exception:

            return None

        if not np.all(
            np.isfinite(coeffs)
        ):

            return None

        return coeffs.astype(
            np.float64
        )

    # ==============================================================
    # EVALUATE X
    # ==============================================================

    def evaluate_x(
        self,
        coeffs,
        y
    ):

        if coeffs is None:
            return np.nan

        a, b = coeffs

        return float(
            a * y + b
        )

    # ==============================================================
    # TEMPORAL GATE
    # ==============================================================

    def temporal_gate(
        self,
        measurement,
        previous,
        image_height,
        image_width,
        is_left
    ):

        if measurement is None:

            return None

        if previous is None:

            return measurement

        y = (
            0.95
            *
            image_height
        )

        current_x = self.evaluate_x(
            measurement,
            y
        )

        previous_x = self.evaluate_x(
            previous,
            y
        )

        x_jump = abs(
            current_x
            -
            previous_x
        )

        slope_change = abs(
            measurement[0]
            -
            previous[0]
        )

        max_x_jump = (
            self.MAX_BOTTOM_X_JUMP_RATIO
            *
            image_width
        )

        # ----------------------------------------------------------
        # IMPORTANT:
        #
        # We no longer hard-reject based on either criterion alone.
        #
        # Only reject a measurement when BOTH position AND slope
        # are wildly inconsistent.
        # ----------------------------------------------------------

        bad_position = (
            x_jump
            >
            max_x_jump
        )

        bad_slope = (
            slope_change
            >
            self.MAX_SLOPE_CHANGE
        )

        if bad_position and bad_slope:

            if is_left:
                self.left_rejected += 1
            else:
                self.right_rejected += 1

            return None

        return measurement

    # ==============================================================
    # UPDATE TRACK
    # ==============================================================

    def update_lane_track(
        self,
        measurement,
        is_left
    ):

        alpha = self.TRACK_ALPHA

        if is_left:

            previous = (
                self.tracked_left_lane
            )

            if measurement is not None:

                measurement = np.asarray(
                    measurement,
                    dtype=np.float64
                )

                if previous is None:

                    tracked = measurement

                else:

                    previous = np.asarray(
                        previous,
                        dtype=np.float64
                    )

                    tracked = (
                        alpha * measurement
                        +
                        (1.0 - alpha)
                        * previous
                    )

                self.tracked_left_lane = tracked

                self.left_missed_frames = 0

                return tracked

            self.left_missed_frames += 1

            if (
                previous is not None
                and
                self.left_missed_frames
                <= self.MAX_MISSED_FRAMES
            ):

                return previous

            self.tracked_left_lane = None

            return None

        # ==========================================================
        # RIGHT
        # ==========================================================

        previous = (
            self.tracked_right_lane
        )

        if measurement is not None:

            measurement = np.asarray(
                measurement,
                dtype=np.float64
            )

            if previous is None:

                tracked = measurement

            else:

                previous = np.asarray(
                    previous,
                    dtype=np.float64
                )

                tracked = (
                    alpha * measurement
                    +
                    (1.0 - alpha)
                    * previous
                )

            self.tracked_right_lane = tracked

            self.right_missed_frames = 0

            return tracked

        self.right_missed_frames += 1

        if (
            previous is not None
            and
            self.right_missed_frames
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
            +
            b
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

        points = np.column_stack(
            [
                x_values[valid],
                y_values[valid]
            ]
        ).astype(
            np.int32
        )

        return points.reshape(
            (-1, 1, 2)
        )

    # ==============================================================
    # CENTERLINE
    # ==============================================================

    def compute_center_points(
        self,
        left_lane,
        right_lane,
        y_values,
        image_width
    ):

        left_x = np.asarray(
            [
                self.evaluate_x(
                    left_lane,
                    y
                )
                for y in y_values
            ]
        )

        right_x = np.asarray(
            [
                self.evaluate_x(
                    right_lane,
                    y
                )
                for y in y_values
            ]
        )

        center_x = (
            left_x
            +
            right_x
        ) / 2.0

        valid = (
            np.isfinite(center_x)
            &
            (center_x >= 0)
            &
            (center_x < image_width)
        )

        if not np.any(valid):

            return None

        points = np.column_stack(
            [
                center_x[valid],
                y_values[valid]
            ]
        ).astype(
            np.int32
        )

        return points.reshape(
            (-1, 1, 2)
        )

    # ==============================================================
    # COMPUTE GEOMETRY
    # ==============================================================

    def compute_lane_geometry(
        self,
        left_lane,
        right_lane,
        image_height,
        image_width
    ):

        # ----------------------------------------------------------
        # Multiple look-ahead levels.
        # ----------------------------------------------------------

        y_values = np.asarray(
            [
                0.60 * image_height,
                0.70 * image_height,
                0.80 * image_height,
                0.90 * image_height,
                0.95 * image_height
            ],
            dtype=np.float64
        )

        left_x = np.asarray(
            [
                self.evaluate_x(
                    left_lane,
                    y
                )
                for y in y_values
            ]
        )

        right_x = np.asarray(
            [
                self.evaluate_x(
                    right_lane,
                    y
                )
                for y in y_values
            ]
        )

        widths = (
            right_x
            -
            left_x
        )

        # ----------------------------------------------------------
        # Keep geometrically meaningful samples.
        # ----------------------------------------------------------

        valid = (
            np.isfinite(left_x)
            &
            np.isfinite(right_x)
            &
            (right_x > left_x)
            &
            (widths >
             self.MIN_LANE_WIDTH_RATIO
             * image_width)
            &
            (widths <
             self.MAX_LANE_WIDTH_RATIO
             * image_width)
        )

        if np.count_nonzero(valid) < 3:

            self.geometry_warnings += 1

            return None

        left_x = left_x[valid]
        right_x = right_x[valid]
        widths = widths[valid]
        y_valid = y_values[valid]

        # ==========================================================
        # CENTER
        # ==========================================================

        center_x = (
            left_x
            +
            right_x
        ) / 2.0

        # ==========================================================
        # BOTTOM VALUES
        # ==========================================================

        bottom_width_raw = float(
            widths[-1]
        )

        bottom_center_raw = float(
            center_x[-1]
        )

        image_center = (
            image_width / 2.0
        )

        offset_raw = (
            bottom_center_raw
            -
            image_center
        )

        # ----------------------------------------------------------
        # Don't allow an absurd center offset.
        # ----------------------------------------------------------

        max_offset = (
            self.MAX_CENTER_OFFSET_RATIO
            *
            image_width
        )

        if abs(offset_raw) > max_offset:

            self.geometry_warnings += 1

            return None

        # ==========================================================
        # HEADING
        # ==========================================================

        # Vehicle forward direction:
        #
        # near -> far
        #
        # Image coordinates:
        #
        # +x = right
        # +y = down
        #
        # We use the centerline vector from near to far.

        x_near = float(
            center_x[-1]
        )

        y_near = float(
            y_valid[-1]
        )

        x_far = float(
            center_x[0]
        )

        y_far = float(
            y_valid[0]
        )

        dx = (
            x_far
            -
            x_near
        )

        dy = (
            y_near
            -
            y_far
        )

        if abs(dy) > 1e-6:

            heading_raw = float(
                np.degrees(
                    np.arctan2(
                        dx,
                        dy
                    )
                )
            )

        else:

            heading_raw = 0.0

        # ==========================================================
        # TEMPORAL GEOMETRY SMOOTHING
        # ==========================================================

        if self.smoothed_width is None:

            self.smoothed_width = (
                bottom_width_raw
            )

        else:

            self.smoothed_width = (
                self.GEOMETRY_ALPHA
                * bottom_width_raw
                +
                (
                    1.0
                    -
                    self.GEOMETRY_ALPHA
                )
                * self.smoothed_width
            )

        if self.smoothed_offset is None:

            self.smoothed_offset = (
                offset_raw
            )

        else:

            self.smoothed_offset = (
                self.GEOMETRY_ALPHA
                * offset_raw
                +
                (
                    1.0
                    -
                    self.GEOMETRY_ALPHA
                )
                * self.smoothed_offset
            )

        if self.smoothed_heading is None:

            self.smoothed_heading = (
                heading_raw
            )

        else:

            # ------------------------------------------------------
            # Heading wrap-safe smoothing.
            # ------------------------------------------------------

            delta = (
                heading_raw
                -
                self.smoothed_heading
            )

            while delta > 180.0:
                delta -= 360.0

            while delta < -180.0:
                delta += 360.0

            self.smoothed_heading += (
                self.GEOMETRY_ALPHA
                * delta
            )

        # ==========================================================
        # CENTERLINE POINTS
        # ==========================================================

        center_points = np.column_stack(
            [
                center_x,
                y_valid
            ]
        ).astype(
            np.int32
        ).reshape(
            (-1, 1, 2)
        )

        # ==========================================================
        # LOOKAHEAD
        # ==========================================================

        lookahead_points = []

        for i in range(
            len(center_x)
        ):

            lookahead_points.append(
                {
                    "x": int(
                        center_x[i]
                    ),
                    "y": int(
                        y_valid[i]
                    ),
                    "width": float(
                        widths[i]
                    )
                }
            )

        # ==========================================================
        # WIDTH STD
        # ==========================================================

        width_std = float(
            np.std(widths)
        )

        return {
            "center_points": center_points,

            "lookahead_points":
                lookahead_points,

            "width":
                float(
                    self.smoothed_width
                ),

            "offset":
                float(
                    self.smoothed_offset
                ),

            "heading":
                float(
                    self.smoothed_heading
                ),

            "width_std":
                width_std
        }


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