#!/usr/bin/env python3

import time

import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy
)

from sensor_msgs.msg import Image

from cv_bridge import CvBridge

from torchvision.models.optical_flow import (
    raft_small,
    Raft_Small_Weights
)


class RAFTOpticalFlowNode(Node):

    def __init__(self):

        super().__init__("raft_optical_flow_node")

        # ==========================================================
        # STARTUP
        # ==========================================================

        self.get_logger().info(
            "=========================================="
        )

        self.get_logger().info(
            "RAFT-Small Optical Flow Node"
        )

        self.get_logger().info(
            "=========================================="
        )

        # ==========================================================
        # DEVICE
        # ==========================================================

        self.device = torch.device("cpu")

        self.get_logger().info(
            f"Device: {self.device}"
        )

        # ==========================================================
        # RAFT INPUT RESOLUTION
        #
        # Keep this resolution for the first benchmark.
        #
        # 640 x 360 is divisible by 8.
        # ==========================================================

        self.INPUT_WIDTH = 640
        self.INPUT_HEIGHT = 360

        self.get_logger().info(
            "RAFT resolution: "
            f"{self.INPUT_WIDTH}x{self.INPUT_HEIGHT}"
        )

        # ==========================================================
        # MODEL
        # ==========================================================

        self.get_logger().info(
            "Loading RAFT-Small..."
        )

        weights = Raft_Small_Weights.DEFAULT

        self.model = raft_small(
            weights=weights
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        self.transforms = (
            weights.transforms()
        )

        self.get_logger().info(
            "RAFT-Small loaded successfully."
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
            "/camera/raft_optical_flow/overlay",
            qos
        )

        self.magnitude_pub = self.create_publisher(
            Image,
            "/camera/raft_optical_flow/magnitude",
            qos
        )

        self.dx_pub = self.create_publisher(
            Image,
            "/camera/raft_optical_flow/dx",
            qos
        )

        self.dy_pub = self.create_publisher(
            Image,
            "/camera/raft_optical_flow/dy",
            qos
        )

        # ==========================================================
        # FRAME STATE
        # ==========================================================

        self.prev_frame = None
        self.prev_timestamp = None

        # ==========================================================
        # STATISTICS
        # ==========================================================

        self.processed_frames = 0

        self.last_inference_time = 0.0

        self.last_dt = 0.0

        self.last_mean = 0.0
        self.last_median = 0.0
        self.last_p75 = 0.0
        self.last_p90 = 0.0
        self.last_p95 = 0.0
        self.last_p99 = 0.0
        self.last_max = 0.0

        self.last_mean_dx = 0.0
        self.last_mean_dy = 0.0

        # ==========================================================
        # SANITY PARAMETERS
        # ==========================================================

        # This is NOT a hard rejection threshold.
        # It is only used to report how much of the field is
        # extremely large.
        self.SANITY_FLOW_THRESHOLD = 200.0

        # ==========================================================
        # VISUALIZATION
        # ==========================================================

        self.ARROW_STEP = 24

        self.ARROW_MIN_MAGNITUDE = 1.0

        self.MAX_VIS_MAGNITUDE = 50.0

        # ==========================================================
        # STARTUP
        # ==========================================================

        self.get_logger().info(
            "Dense optical flow enabled."
        )

        self.get_logger().info(
            "Robust flow statistics enabled."
        )

        self.get_logger().info(
            "Timestamp / delta-time monitoring enabled."
        )

        self.get_logger().info(
            "Waiting for /camera/front/image ..."
        )

    # ==============================================================
    # IMAGE CALLBACK
    # ==============================================================

    def image_callback(self, msg):

        try:

            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

        except Exception as e:

            self.get_logger().error(
                f"Image conversion failed: {e}"
            )

            return

        # ==========================================================
        # CURRENT ROS TIMESTAMP
        # ==========================================================

        current_timestamp = (
            float(msg.header.stamp.sec)
            +
            float(msg.header.stamp.nanosec)
            *
            1e-9
        )

        # ==========================================================
        # FIRST FRAME
        # ==========================================================

        if self.prev_frame is None:

            self.prev_frame = frame.copy()

            self.prev_timestamp = (
                current_timestamp
            )

            self.get_logger().info(
                "Initial frame received. "
                "Waiting for next frame..."
            )

            return

        # ==========================================================
        # DELTA TIME
        # ==========================================================

        dt = (
            current_timestamp
            -
            self.prev_timestamp
        )

        self.last_dt = dt

        # ==========================================================
        # CHECK TIMESTAMP
        # ==========================================================

        if dt <= 0.0:

            self.get_logger().warn(
                f"Invalid timestamp difference: "
                f"dt={dt:.6f}s"
            )

            self.prev_frame = frame.copy()

            self.prev_timestamp = (
                current_timestamp
            )

            return

        # ==========================================================
        # COMPUTE RAFT FLOW
        # ==========================================================

        flow, inference_time = (
            self.compute_flow(
                self.prev_frame,
                frame
            )
        )

        if flow is None:

            self.prev_frame = frame.copy()

            self.prev_timestamp = (
                current_timestamp
            )

            return

        # ==========================================================
        # FLOW COMPONENTS
        # ==========================================================

        dx = flow[:, :, 0]

        dy = flow[:, :, 1]

        magnitude = np.sqrt(
            dx * dx
            +
            dy * dy
        )

        # ==========================================================
        # STATISTICS
        # ==========================================================

        self.update_statistics(
            dx,
            dy,
            magnitude
        )

        self.last_inference_time = (
            inference_time
        )

        # ==========================================================
        # VISUALIZATION
        # ==========================================================

        overlay = (
            self.create_flow_visualization(
                frame,
                flow
            )
        )

        magnitude_image = (
            self.create_magnitude_image(
                magnitude
            )
        )

        dx_image = (
            self.create_component_image(
                dx
            )
        )

        dy_image = (
            self.create_component_image(
                dy
            )
        )

        # ==========================================================
        # PUBLISH
        # ==========================================================

        self.publish_outputs(
            overlay,
            magnitude_image,
            dx_image,
            dy_image,
            msg
        )

        # ==========================================================
        # UPDATE PREVIOUS FRAME
        # ==========================================================

        self.prev_frame = frame.copy()

        self.prev_timestamp = (
            current_timestamp
        )

        self.processed_frames += 1

        # ==========================================================
        # LOG
        # ==========================================================

        if self.processed_frames % 5 == 0:

            fps = (
                1.0 / inference_time
                if inference_time > 0.0
                else 0.0
            )

            high_flow_percentage = (
                100.0
                *
                np.mean(
                    magnitude
                    >
                    self.SANITY_FLOW_THRESHOLD
                )
            )

            self.get_logger().info(
                "RAFT flow | "
                f"processed={self.processed_frames} | "
                f"dt={dt:.3f}s | "
                f"inference={inference_time:.2f}s | "
                f"FPS={fps:.2f} | "
                f"mean={self.last_mean:.2f}px | "
                f"median={self.last_median:.2f}px | "
                f"P90={self.last_p90:.2f}px | "
                f"P95={self.last_p95:.2f}px | "
                f"P99={self.last_p99:.2f}px | "
                f"max={self.last_max:.2f}px | "
                f">200px={high_flow_percentage:.1f}%"
            )

    # ==============================================================
    # COMPUTE RAFT FLOW
    # ==============================================================

    def compute_flow(
        self,
        previous_frame,
        current_frame
    ):

        # ==========================================================
        # RESIZE BOTH FRAMES
        # ==========================================================

        previous_resized = cv2.resize(
            previous_frame,
            (
                self.INPUT_WIDTH,
                self.INPUT_HEIGHT
            ),
            interpolation=cv2.INTER_AREA
        )

        current_resized = cv2.resize(
            current_frame,
            (
                self.INPUT_WIDTH,
                self.INPUT_HEIGHT
            ),
            interpolation=cv2.INTER_AREA
        )

        # ==========================================================
        # BGR → RGB
        # ==========================================================

        previous_rgb = cv2.cvtColor(
            previous_resized,
            cv2.COLOR_BGR2RGB
        )

        current_rgb = cv2.cvtColor(
            current_resized,
            cv2.COLOR_BGR2RGB
        )

        # ==========================================================
        # NUMPY → TORCH
        # ==========================================================

        previous_tensor = (
            torch.from_numpy(
                previous_rgb
            )
            .permute(
                2,
                0,
                1
            )
            .float()
            .unsqueeze(0)
        )

        current_tensor = (
            torch.from_numpy(
                current_rgb
            )
            .permute(
                2,
                0,
                1
            )
            .float()
            .unsqueeze(0)
        )

        # ==========================================================
        # TORCHVISION RAFT TRANSFORM
        # ==========================================================

        (
            previous_tensor,
            current_tensor
        ) = self.transforms(
            previous_tensor,
            current_tensor
        )

        previous_tensor = (
            previous_tensor.to(
                self.device
            )
        )

        current_tensor = (
            current_tensor.to(
                self.device
            )
        )

        # ==========================================================
        # INFERENCE
        # ==========================================================

        start = time.perf_counter()

        with torch.inference_mode():

            predictions = self.model(
                previous_tensor,
                current_tensor
            )

        end = time.perf_counter()

        inference_time = (
            end - start
        )

        # ==========================================================
        # FINAL FLOW PREDICTION
        # ==========================================================

        flow_tensor = predictions[-1]

        # Shape:
        #
        # [1, 2, H, W]
        #
        # H = 360
        # W = 640

        flow = (
            flow_tensor[0]
            .permute(
                1,
                2,
                0
            )
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        # ==========================================================
        # IMPORTANT
        #
        # DO NOT RESIZE OR SCALE THE FLOW.
        #
        # The returned flow remains in the RAFT working
        # coordinate system:
        #
        # 640 x 360
        #
        # This makes the measurements internally consistent.
        # ==========================================================

        return (
            flow,
            inference_time
        )

    # ==============================================================
    # STATISTICS
    # ==============================================================

    def update_statistics(
        self,
        dx,
        dy,
        magnitude
    ):

        finite_mask = (
            np.isfinite(dx)
            &
            np.isfinite(dy)
            &
            np.isfinite(magnitude)
        )

        values = (
            magnitude[
                finite_mask
            ]
        )

        dx_values = (
            dx[
                finite_mask
            ]
        )

        dy_values = (
            dy[
                finite_mask
            ]
        )

        if len(values) == 0:

            return

        self.last_mean = float(
            np.mean(values)
        )

        self.last_median = float(
            np.median(values)
        )

        self.last_p75 = float(
            np.percentile(
                values,
                75
            )
        )

        self.last_p90 = float(
            np.percentile(
                values,
                90
            )
        )

        self.last_p95 = float(
            np.percentile(
                values,
                95
            )
        )

        self.last_p99 = float(
            np.percentile(
                values,
                99
            )
        )

        self.last_max = float(
            np.max(values)
        )

        self.last_mean_dx = float(
            np.mean(dx_values)
        )

        self.last_mean_dy = float(
            np.mean(dy_values)
        )

    # ==============================================================
    # FLOW VISUALIZATION
    # ==============================================================

    def create_flow_visualization(
        self,
        frame,
        flow
    ):

        # ----------------------------------------------------------
        # Work entirely at RAFT resolution
        # ----------------------------------------------------------

        small = cv2.resize(
            frame,
            (
                self.INPUT_WIDTH,
                self.INPUT_HEIGHT
            ),
            interpolation=cv2.INTER_AREA
        )

        dx = flow[:, :, 0]

        dy = flow[:, :, 1]

        magnitude = np.sqrt(
            dx * dx
            +
            dy * dy
        )

        angle = np.arctan2(
            dy,
            dx
        )

        # ==========================================================
        # HSV FLOW VISUALIZATION
        # ==========================================================

        hsv = np.zeros(
            (
                self.INPUT_HEIGHT,
                self.INPUT_WIDTH,
                3
            ),
            dtype=np.uint8
        )

        # Direction → Hue

        hue = (
            (
                angle
                +
                np.pi
            )
            *
            90.0
            /
            np.pi
        )

        hsv[:, :, 0] = (
            np.mod(
                hue,
                180
            )
            .astype(
                np.uint8
            )
        )

        # Saturation

        hsv[:, :, 1] = 255

        # Magnitude → Value

        magnitude_clipped = np.clip(
            magnitude,
            0.0,
            self.MAX_VIS_MAGNITUDE
        )

        hsv[:, :, 2] = (
            magnitude_clipped
            /
            self.MAX_VIS_MAGNITUDE
            *
            255.0
        ).astype(
            np.uint8
        )

        flow_color = cv2.cvtColor(
            hsv,
            cv2.COLOR_HSV2BGR
        )

        # ==========================================================
        # BLEND
        # ==========================================================

        overlay = cv2.addWeighted(
            small,
            0.50,
            flow_color,
            0.50,
            0
        )

        # ==========================================================
        # DRAW FLOW ARROWS
        # ==========================================================

        for y in range(
            self.ARROW_STEP // 2,
            self.INPUT_HEIGHT,
            self.ARROW_STEP
        ):

            for x in range(
                self.ARROW_STEP // 2,
                self.INPUT_WIDTH,
                self.ARROW_STEP
            ):

                fx = float(
                    dx[y, x]
                )

                fy = float(
                    dy[y, x]
                )

                mag = float(
                    magnitude[y, x]
                )

                if (
                    not np.isfinite(mag)
                    or
                    mag < self.ARROW_MIN_MAGNITUDE
                ):

                    continue

                end_x = int(
                    np.clip(
                        x + fx,
                        0,
                        self.INPUT_WIDTH - 1
                    )
                )

                end_y = int(
                    np.clip(
                        y + fy,
                        0,
                        self.INPUT_HEIGHT - 1
                    )
                )

                cv2.arrowedLine(
                    overlay,
                    (x, y),
                    (end_x, end_y),
                    (0, 255, 0),
                    1,
                    tipLength=0.25
                )

        # ==========================================================
        # TEXT
        # ==========================================================

        cv2.putText(
            overlay,
            "RAFT-Small Dense Optical Flow",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        cv2.putText(
            overlay,
            (
                f"dt={self.last_dt:.3f}s"
            ),
            (15, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

        cv2.putText(
            overlay,
            (
                f"inference="
                f"{self.last_inference_time:.2f}s"
            ),
            (15, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

        cv2.putText(
            overlay,
            (
                f"median="
                f"{self.last_median:.1f}px"
            ),
            (15, 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

        cv2.putText(
            overlay,
            (
                f"P95="
                f"{self.last_p95:.1f}px"
            ),
            (15, 116),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

        return overlay

    # ==============================================================
    # MAGNITUDE IMAGE
    # ==============================================================

    def create_magnitude_image(
        self,
        magnitude
    ):

        magnitude_clipped = np.clip(
            magnitude,
            0.0,
            self.MAX_VIS_MAGNITUDE
        )

        normalized = (
            magnitude_clipped
            /
            self.MAX_VIS_MAGNITUDE
            *
            255.0
        ).astype(
            np.uint8
        )

        return cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_JET
        )

    # ==============================================================
    # COMPONENT IMAGE
    # ==============================================================

    def create_component_image(
        self,
        component
    ):

        # Symmetric range for visualization.

        component_clipped = np.clip(
            component,
            -50.0,
            50.0
        )

        normalized = (
            (
                component_clipped
                +
                50.0
            )
            /
            100.0
            *
            255.0
        ).astype(
            np.uint8
        )

        return cv2.applyColorMap(
            normalized,
            cv2.COLORMAP_JET
        )

    # ==============================================================
    # PUBLISH
    # ==============================================================

    def publish_outputs(
        self,
        overlay,
        magnitude,
        dx_image,
        dy_image,
        msg
    ):

        try:

            overlay_msg = (
                self.bridge.cv2_to_imgmsg(
                    overlay,
                    encoding="bgr8"
                )
            )

            overlay_msg.header = (
                msg.header
            )

            self.overlay_pub.publish(
                overlay_msg
            )

            magnitude_msg = (
                self.bridge.cv2_to_imgmsg(
                    magnitude,
                    encoding="bgr8"
                )
            )

            magnitude_msg.header = (
                msg.header
            )

            self.magnitude_pub.publish(
                magnitude_msg
            )

            dx_msg = (
                self.bridge.cv2_to_imgmsg(
                    dx_image,
                    encoding="bgr8"
                )
            )

            dx_msg.header = (
                msg.header
            )

            self.dx_pub.publish(
                dx_msg
            )

            dy_msg = (
                self.bridge.cv2_to_imgmsg(
                    dy_image,
                    encoding="bgr8"
                )
            )

            dy_msg.header = (
                msg.header
            )

            self.dy_pub.publish(
                dy_msg
            )

        except Exception as e:

            self.get_logger().error(
                f"Failed to publish flow: {e}"
            )


# ==================================================================
# MAIN
# ==================================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = RAFTOpticalFlowNode()

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