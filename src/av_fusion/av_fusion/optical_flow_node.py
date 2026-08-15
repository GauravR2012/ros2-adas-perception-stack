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


class OpticalFlowNode(Node):

    def __init__(self):

        super().__init__("optical_flow_node")

        # ==========================================================
        # STARTUP
        # ==========================================================

        self.get_logger().info(
            "=========================================="
        )

        self.get_logger().info(
            "Sparse Optical Flow Node"
        )

        self.get_logger().info(
            "=========================================="
        )

        self.get_logger().info(
            "Method: Shi-Tomasi + Lucas-Kanade"
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

        self.image_sub = self.create_subscription(
            Image,
            "/camera/front/image",
            self.image_callback,
            qos
        )

        # ==========================================================
        # PUBLISHERS
        # ==========================================================

        self.overlay_pub = self.create_publisher(
            Image,
            "/camera/optical_flow/overlay",
            qos
        )

        self.flow_mask_pub = self.create_publisher(
            Image,
            "/camera/optical_flow/mask",
            qos
        )

        # ==========================================================
        # PREVIOUS FRAME
        # ==========================================================

        self.prev_gray = None

        self.prev_points = None

        self.prev_header = None

        # ==========================================================
        # FEATURE PARAMETERS
        # ==========================================================

        self.MAX_CORNERS = 300

        self.QUALITY_LEVEL = 0.01

        self.MIN_DISTANCE = 15

        self.BLOCK_SIZE = 7

        # ==========================================================
        # LUCAS-KANADE PARAMETERS
        # ==========================================================

        self.LK_WIN_SIZE = (
            21,
            21
        )

        self.LK_MAX_LEVEL = 3

        self.LK_CRITERIA = (
            cv2.TERM_CRITERIA_EPS
            |
            cv2.TERM_CRITERIA_COUNT,
            30,
            0.01
        )

        # ==========================================================
        # FEATURE RE-DETECTION
        # ==========================================================

        self.MIN_TRACKED_POINTS = 80

        self.REDETECT_INTERVAL = 5

        # ==========================================================
        # OUTLIER FILTERING
        # ==========================================================

        self.MAX_FLOW_MAGNITUDE = 80.0

        self.MIN_FLOW_MAGNITUDE = 0.05

        # ==========================================================
        # ROI
        # ==========================================================

        self.USE_ROI = True

        self.ROI_TOP_RATIO = 0.35

        # ==========================================================
        # VISUALIZATION
        # ==========================================================

        self.ARROW_SCALE = 2.0

        self.LINE_THICKNESS = 2

        self.POINT_RADIUS = 3

        # ==========================================================
        # STATISTICS
        # ==========================================================

        self.processed_frames = 0

        self.total_features_detected = 0

        self.total_tracks = 0

        self.last_mean_flow = 0.0

        self.last_median_flow = 0.0

        self.last_max_flow = 0.0

        # ==========================================================
        # STARTUP LOG
        # ==========================================================

        self.get_logger().info(
            "Shi-Tomasi feature detection enabled."
        )

        self.get_logger().info(
            "Pyramidal Lucas-Kanade tracking enabled."
        )

        self.get_logger().info(
            f"Maximum corners: {self.MAX_CORNERS}"
        )

        self.get_logger().info(
            f"Minimum tracked points: "
            f"{self.MIN_TRACKED_POINTS}"
        )

        self.get_logger().info(
            "Waiting for camera..."
        )

    # ==============================================================
    # CAMERA CALLBACK
    # ==============================================================

    def image_callback(self, msg):

        try:

            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

        except Exception as e:

            self.get_logger().error(
                f"Failed to convert image: {e}"
            )

            return

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # ==========================================================
        # FIRST FRAME
        # ==========================================================

        if self.prev_gray is None:

            self.prev_gray = gray

            self.prev_points = (
                self.detect_features(
                    gray
                )
            )

            self.prev_header = msg.header

            self.processed_frames += 1

            self.get_logger().info(
                f"Initial frame | "
                f"features={self.get_feature_count()}"
            )

            return

        # ==========================================================
        # TRACK FEATURES
        # ==========================================================

        current_points = None

        if (
            self.prev_points is not None
            and
            len(self.prev_points) > 0
        ):

            current_points, status, error = (
                self.track_features(
                    self.prev_gray,
                    gray,
                    self.prev_points
                )
            )

        # ==========================================================
        # NO FEATURES
        # ==========================================================

        if (
            current_points is None
            or
            len(current_points) == 0
        ):

            current_points = (
                self.detect_features(
                    gray
                )
            )

            self.prev_gray = gray

            self.prev_points = current_points

            self.publish_empty_output(
                frame,
                msg
            )

            self.processed_frames += 1

            return

        # ==========================================================
        # FILTER FLOW
        # ==========================================================

        (
            good_prev,
            good_curr,
            magnitudes
        ) = self.filter_flow(
            self.prev_points,
            current_points
        )

        # ==========================================================
        # VISUALIZATION
        # ==========================================================

        overlay = frame.copy()

        flow_mask = np.zeros(
            (
                frame.shape[0],
                frame.shape[1]
            ),
            dtype=np.uint8
        )

        self.draw_flow(
            overlay,
            flow_mask,
            good_prev,
            good_curr,
            magnitudes
        )

        # ==========================================================
        # STATISTICS
        # ==========================================================

        if len(magnitudes) > 0:

            self.last_mean_flow = float(
                np.mean(magnitudes)
            )

            self.last_median_flow = float(
                np.median(magnitudes)
            )

            self.last_max_flow = float(
                np.max(magnitudes)
            )

        else:

            self.last_mean_flow = 0.0

            self.last_median_flow = 0.0

            self.last_max_flow = 0.0

        # ==========================================================
        # FEATURE RE-DETECTION
        # ==========================================================

        should_redetect = (
            len(good_curr)
            <
            self.MIN_TRACKED_POINTS
        )

        should_redetect = (
            should_redetect
            or
            (
                self.processed_frames
                %
                self.REDETECT_INTERVAL
                ==
                0
            )
        )

        if should_redetect:

            new_points = (
                self.detect_features(
                    gray
                )
            )

            if new_points is not None:

                if len(good_curr) > 0:

                    tracked = (
                        good_curr.reshape(
                            -1,
                            1,
                            2
                        )
                    )

                    if len(tracked) > 0:

                        new_points = (
                            self.remove_nearby_points(
                                new_points,
                                tracked
                            )
                        )

                if new_points is not None:

                    if len(good_curr) > 0:

                        combined = np.vstack(
                            [
                                good_curr.reshape(
                                    -1,
                                    1,
                                    2
                                ),
                                new_points
                            ]
                        )

                    else:

                        combined = new_points

                    if len(combined) > self.MAX_CORNERS:

                        combined = combined[
                            :self.MAX_CORNERS
                        ]

                    self.prev_points = (
                        combined.astype(
                            np.float32
                        )
                    )

                else:

                    self.prev_points = (
                        good_curr.reshape(
                            -1,
                            1,
                            2
                        ).astype(
                            np.float32
                        )
                    )

            else:

                self.prev_points = (
                    good_curr.reshape(
                        -1,
                        1,
                        2
                    ).astype(
                        np.float32
                    )
                )

        else:

            self.prev_points = (
                good_curr.reshape(
                    -1,
                    1,
                    2
                ).astype(
                    np.float32
                )
            )

        # ==========================================================
        # UPDATE PREVIOUS FRAME
        # ==========================================================

        self.prev_gray = gray

        self.prev_header = msg.header

        # ==========================================================
        # DEBUG TEXT
        # ==========================================================

        self.draw_statistics(
            overlay,
            len(good_prev),
            len(good_curr)
        )

        # ==========================================================
        # PUBLISH
        # ==========================================================

        self.publish_outputs(
            overlay,
            flow_mask,
            msg
        )

        # ==========================================================
        # FRAME COUNT
        # ==========================================================

        self.processed_frames += 1

        if self.processed_frames % 20 == 0:

            self.get_logger().info(
                "Optical flow | "
                f"processed={self.processed_frames} | "
                f"tracked={len(good_curr)} | "
                f"mean={self.last_mean_flow:.2f}px | "
                f"median={self.last_median_flow:.2f}px | "
                f"max={self.last_max_flow:.2f}px"
            )

    # ==============================================================
    # FEATURE DETECTION
    # ==============================================================

    def detect_features(self, gray):

        mask = None

        if self.USE_ROI:

            h, w = gray.shape

            mask = np.zeros(
                (h, w),
                dtype=np.uint8
            )

            top = int(
                self.ROI_TOP_RATIO * h
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
                        int(0.80 * w),
                        top
                    ],
                    [
                        int(0.20 * w),
                        top
                    ]
                ],
                dtype=np.int32
            )

            cv2.fillPoly(
                mask,
                [polygon],
                255
            )

        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.MAX_CORNERS,
            qualityLevel=self.QUALITY_LEVEL,
            minDistance=self.MIN_DISTANCE,
            blockSize=self.BLOCK_SIZE,
            mask=mask
        )

        if points is None:

            return np.empty(
                (0, 1, 2),
                dtype=np.float32
            )

        self.total_features_detected += (
            len(points)
        )

        return points.astype(
            np.float32
        )

    # ==============================================================
    # LUCAS-KANADE
    # ==============================================================

    def track_features(
        self,
        previous_gray,
        current_gray,
        previous_points
    ):

        try:

            current_points, status, error = (
                cv2.calcOpticalFlowPyrLK(
                    previous_gray,
                    current_gray,
                    previous_points,
                    None,
                    winSize=self.LK_WIN_SIZE,
                    maxLevel=self.LK_MAX_LEVEL,
                    criteria=self.LK_CRITERIA
                )
            )

        except Exception as e:

            self.get_logger().error(
                f"Lucas-Kanade failed: {e}"
            )

            return None, None, None

        if (
            current_points is None
            or
            status is None
        ):

            return None, None, None

        status = status.reshape(-1)

        valid = (
            status == 1
        )

        previous_good = (
            previous_points[valid]
        )

        current_good = (
            current_points[valid]
        )

        return (
            current_good,
            status,
            error
        )

    # ==============================================================
    # FLOW FILTER
    # ==============================================================

    def filter_flow(
        self,
        previous_points,
        current_points
    ):

        if (
            previous_points is None
            or
            current_points is None
        ):

            return (
                np.empty((0, 2)),
                np.empty((0, 2)),
                np.empty((0,))
            )

        previous_points = (
            previous_points.reshape(
                -1,
                2
            )
        )

        current_points = (
            current_points.reshape(
                -1,
                2
            )
        )

        n = min(
            len(previous_points),
            len(current_points)
        )

        if n == 0:

            return (
                np.empty((0, 2)),
                np.empty((0, 2)),
                np.empty((0,))
            )

        previous_points = (
            previous_points[:n]
        )

        current_points = (
            current_points[:n]
        )

        displacement = (
            current_points
            -
            previous_points
        )

        magnitudes = np.linalg.norm(
            displacement,
            axis=1
        )

        valid = (
            np.isfinite(magnitudes)
            &
            (
                magnitudes
                >=
                self.MIN_FLOW_MAGNITUDE
            )
            &
            (
                magnitudes
                <=
                self.MAX_FLOW_MAGNITUDE
            )
        )

        good_previous = (
            previous_points[valid]
        )

        good_current = (
            current_points[valid]
        )

        good_magnitudes = (
            magnitudes[valid]
        )

        return (
            good_previous,
            good_current,
            good_magnitudes
        )

    # ==============================================================
    # DRAW FLOW
    # ==============================================================

    def draw_flow(
        self,
        overlay,
        flow_mask,
        previous_points,
        current_points,
        magnitudes
    ):

        for (
            previous,
            current,
            magnitude
        ) in zip(
            previous_points,
            current_points,
            magnitudes
        ):

            x1 = int(
                previous[0]
            )

            y1 = int(
                previous[1]
            )

            x2 = int(
                current[0]
            )

            y2 = int(
                current[1]
            )

            # ------------------------------------------------------
            # Motion vector
            # ------------------------------------------------------

            cv2.arrowedLine(
                overlay,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                self.LINE_THICKNESS,
                tipLength=0.25
            )

            # ------------------------------------------------------
            # Current point
            # ------------------------------------------------------

            cv2.circle(
                overlay,
                (x2, y2),
                self.POINT_RADIUS,
                (0, 0, 255),
                -1
            )

            # ------------------------------------------------------
            # Previous point in flow mask
            # ------------------------------------------------------

            cv2.line(
                flow_mask,
                (x1, y1),
                (x2, y2),
                255,
                2
            )

            cv2.circle(
                flow_mask,
                (x2, y2),
                2,
                255,
                -1
            )

    # ==============================================================
    # REMOVE NEARBY FEATURES
    # ==============================================================

    def remove_nearby_points(
        self,
        new_points,
        tracked_points
    ):

        if (
            new_points is None
            or
            len(new_points) == 0
        ):

            return new_points

        if (
            tracked_points is None
            or
            len(tracked_points) == 0
        ):

            return new_points

        new_xy = (
            new_points.reshape(
                -1,
                2
            )
        )

        tracked_xy = (
            tracked_points.reshape(
                -1,
                2
            )
        )

        keep = []

        for point in new_xy:

            distances = np.linalg.norm(
                tracked_xy
                -
                point,
                axis=1
            )

            if np.min(distances) > (
                self.MIN_DISTANCE
            ):

                keep.append(
                    point
                )

        if len(keep) == 0:

            return np.empty(
                (0, 1, 2),
                dtype=np.float32
            )

        return np.asarray(
            keep,
            dtype=np.float32
        ).reshape(
            -1,
            1,
            2
        )

    # ==============================================================
    # STATISTICS
    # ==============================================================

    def draw_statistics(
        self,
        overlay,
        previous_count,
        current_count
    ):

        cv2.putText(
            overlay,
            "Sparse Optical Flow",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            (0, 255, 255),
            2
        )

        cv2.putText(
            overlay,
            (
                f"Tracked: "
                f"{current_count}"
            ),
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            overlay,
            (
                f"Mean flow: "
                f"{self.last_mean_flow:.2f}px"
            ),
            (30, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        cv2.putText(
            overlay,
            (
                f"Median flow: "
                f"{self.last_median_flow:.2f}px"
            ),
            (30, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        cv2.putText(
            overlay,
            (
                f"Max flow: "
                f"{self.last_max_flow:.2f}px"
            ),
            (30, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 180, 0),
            2
        )

        cv2.putText(
            overlay,
            (
                f"Frame: "
                f"{self.processed_frames}"
            ),
            (30, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    # ==============================================================
    # PUBLISH OUTPUTS
    # ==============================================================

    def publish_outputs(
        self,
        overlay,
        flow_mask,
        msg
    ):

        try:

            overlay_msg = (
                self.bridge.cv2_to_imgmsg(
                    overlay,
                    encoding="bgr8"
                )
            )

            overlay_msg.header = msg.header

            self.overlay_pub.publish(
                overlay_msg
            )

            mask_msg = (
                self.bridge.cv2_to_imgmsg(
                    flow_mask,
                    encoding="mono8"
                )
            )

            mask_msg.header = msg.header

            self.flow_mask_pub.publish(
                mask_msg
            )

        except Exception as e:

            self.get_logger().error(
                f"Failed to publish optical flow: {e}"
            )

    # ==============================================================
    # EMPTY OUTPUT
    # ==============================================================

    def publish_empty_output(
        self,
        frame,
        msg
    ):

        overlay = frame.copy()

        flow_mask = np.zeros(
            (
                frame.shape[0],
                frame.shape[1]
            ),
            dtype=np.uint8
        )

        cv2.putText(
            overlay,
            "Optical Flow: waiting for features",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 165, 255),
            2
        )

        self.publish_outputs(
            overlay,
            flow_mask,
            msg
        )

    # ==============================================================
    # FEATURE COUNT
    # ==============================================================

    def get_feature_count(self):

        if self.prev_points is None:
            return 0

        return len(
            self.prev_points
        )


# ==================================================================
# MAIN
# ==================================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = OpticalFlowNode()

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
