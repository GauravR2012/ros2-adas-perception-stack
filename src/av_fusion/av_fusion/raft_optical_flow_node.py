#!/usr/bin/env python3

import threading
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
)

from sensor_msgs.msg import Image

from cv_bridge import CvBridge

import torch

from torchvision.models.optical_flow import (
    raft_small,
    Raft_Small_Weights,
)


class RAFTOpticalFlowNode(Node):

    def __init__(self):

        super().__init__(
            "raft_optical_flow_node"
        )

        # ==========================================================
        # CONFIGURATION
        # ==========================================================

        self.INPUT_WIDTH = 640
        self.INPUT_HEIGHT = 360

        self.MAX_FLOW = 2000.0

        # ==========================================================
        # DEVICE
        # ==========================================================

        self.device = torch.device(
            "cpu"
        )

        # ==========================================================
        # ROS
        # ==========================================================

        self.bridge = CvBridge()

        # IMPORTANT:
        #
        # Depth = 1 prevents ROS from building a large queue
        # of stale camera frames.
        #
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )

        # ==========================================================
        # SUBSCRIBER
        # ==========================================================

        self.image_sub = self.create_subscription(
            Image,
            "/camera/front/image",
            self.image_callback,
            qos,
        )

        # ==========================================================
        # PUBLISHERS
        # ==========================================================

        self.flow_overlay_pub = (
            self.create_publisher(
                Image,
                "/camera/optical_flow/overlay",
                qos,
            )
        )

        self.flow_magnitude_pub = (
            self.create_publisher(
                Image,
                "/camera/optical_flow/magnitude",
                qos,
            )
        )

        # ==========================================================
        # LOAD RAFT
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

        self.get_logger().info(
            f"Device: {self.device}"
        )

        self.get_logger().info(
            f"RAFT resolution: "
            f"{self.INPUT_WIDTH}x"
            f"{self.INPUT_HEIGHT}"
        )

        self.get_logger().info(
            "Loading RAFT-Small..."
        )

        weights = (
            Raft_Small_Weights.DEFAULT
        )

        self.model = raft_small(
            weights=weights
        )

        self.model = (
            self.model.to(
                self.device
            )
        )

        self.model.eval()

        self.get_logger().info(
            "RAFT-Small loaded successfully."
        )

        # ==========================================================
        # THREADING
        # ==========================================================

        # Camera callback NEVER performs RAFT inference.
        #
        # It only stores the newest frame.
        #
        self.latest_frame = None

        self.latest_stamp = None

        self.frame_lock = (
            threading.Lock()
        )

        self.worker_condition = (
            threading.Condition(
                self.frame_lock
            )
        )

        self.worker_running = True

        self.worker_busy = False

        self.worker_thread = (
            threading.Thread(
                target=self.flow_worker,
                daemon=True,
            )
        )

        self.worker_thread.start()

        # ==========================================================
        # FRAME STATE
        # ==========================================================

        self.previous_frame = None

        self.previous_stamp = None

        # ==========================================================
        # STATISTICS
        # ==========================================================

        self.received = 0

        self.processed = 0

        self.dropped = 0

        self.scene_resets = 0

        self.invalid_dt = 0

        self.last_log_time = (
            time.perf_counter()
        )

        # ==========================================================
        # STARTUP
        # ==========================================================

        self.get_logger().info(
            "Dense optical flow enabled."
        )

        self.get_logger().info(
            "Latest-frame buffering enabled."
        )

        self.get_logger().info(
            "Stale-frame dropping enabled."
        )

        self.get_logger().info(
            "Scene-loop timestamp reset enabled."
        )

        self.get_logger().info(
            "Waiting for "
            "/camera/front/image ..."
        )

    # ==============================================================
    # CAMERA CALLBACK
    # ==============================================================

    def image_callback(
        self,
        msg,
    ):

        self.received += 1

        # ==========================================================
        # CONVERT IMAGE
        # ==========================================================

        try:

            frame = (
                self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding="bgr8",
                )
            )

        except Exception as e:

            self.get_logger().error(
                f"Image conversion failed: {e}"
            )

            return

        # ==========================================================
        # COPY
        # ==========================================================

        frame = frame.copy()

        # ==========================================================
        # LATEST FRAME ONLY
        # ==========================================================

        with self.worker_condition:

            # If a frame is already waiting and has not been
            # processed, discard it.
            #
            # We do NOT want:
            #
            # frame 1
            # frame 2
            # frame 3
            # frame 4
            #
            # waiting behind a 3-second RAFT inference.
            #
            if (
                self.latest_frame
                is not None
            ):

                self.dropped += 1

            self.latest_frame = frame

            self.latest_stamp = (
                msg.header.stamp
            )

            self.worker_condition.notify()

    # ==============================================================
    # WORKER
    # ==============================================================

    def flow_worker(self):

        self.get_logger().info(
            "RAFT worker thread started."
        )

        while self.worker_running:

            # ======================================================
            # WAIT
            # ======================================================

            with self.worker_condition:

                while (
                    self.latest_frame is None
                    and
                    self.worker_running
                ):

                    self.worker_condition.wait(
                        timeout=0.5
                    )

                if not self.worker_running:

                    break

                frame = (
                    self.latest_frame
                )

                stamp = (
                    self.latest_stamp
                )

                # IMPORTANT:
                #
                # Remove it from the buffer BEFORE inference.
                #
                self.latest_frame = None

            # ======================================================
            # PROCESS
            # ======================================================

            try:

                self.process_frame(
                    frame,
                    stamp,
                )

            except Exception as e:

                self.get_logger().error(
                    "RAFT processing error: "
                    f"{e}"
                )

        self.get_logger().info(
            "RAFT worker thread stopped."
        )

    # ==============================================================
    # PROCESS FRAME
    # ==============================================================

    def process_frame(
        self,
        frame,
        stamp,
    ):

        # ==========================================================
        # TIMESTAMP
        # ==========================================================

        current_time = (
            stamp.sec
            +
            stamp.nanosec / 1e9
        )

        # ==========================================================
        # INITIAL FRAME
        # ==========================================================

        if (
            self.previous_frame
            is None
        ):

            self.previous_frame = (
                frame
            )

            self.previous_stamp = (
                current_time
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
            current_time
            -
            self.previous_stamp
        )

        # ==========================================================
        # NUSCENES SCENE LOOP
        # ==========================================================

        # When the NuScenes player reaches the end of the scene,
        # it jumps from the final timestamp back to the first
        # timestamp.
        #
        # Example:
        #
        # previous = 153.5s
        # current  = 137.5s
        #
        # dt = -16s
        #
        # That is NOT a physical camera motion.
        #
        if dt <= 0.0:

            self.scene_resets += 1

            self.previous_frame = (
                frame
            )

            self.previous_stamp = (
                current_time
            )

            self.get_logger().warn(
                "NuScenes scene loop detected. "
                f"Resetting temporal flow state "
                f"(dt={dt:.3f}s)."
            )

            return

        # ==========================================================
        # EXTREMELY LARGE DT
        # ==========================================================

        # This should normally not happen after latest-frame
        # buffering, but protect against it.
        #
        if dt > 2.0:

            self.invalid_dt += 1

            self.get_logger().warn(
                "Large timestamp gap detected: "
                f"dt={dt:.3f}s. "
                "Resetting flow state."
            )

            self.previous_frame = (
                frame
            )

            self.previous_stamp = (
                current_time
            )

            return

        # ==========================================================
        # PREVIOUS / CURRENT
        # ==========================================================

        prev = (
            self.previous_frame
        )

        curr = frame

        # ==========================================================
        # UPDATE STATE
        # ==========================================================

        self.previous_frame = (
            frame
        )

        self.previous_stamp = (
            current_time
        )

        # ==========================================================
        # RESIZE
        # ==========================================================

        prev_small = cv2.resize(
            prev,
            (
                self.INPUT_WIDTH,
                self.INPUT_HEIGHT,
            ),
            interpolation=cv2.INTER_AREA,
        )

        curr_small = cv2.resize(
            curr,
            (
                self.INPUT_WIDTH,
                self.INPUT_HEIGHT,
            ),
            interpolation=cv2.INTER_AREA,
        )

        # ==========================================================
        # BGR → RGB
        # ==========================================================

        prev_rgb = cv2.cvtColor(
            prev_small,
            cv2.COLOR_BGR2RGB,
        )

        curr_rgb = cv2.cvtColor(
            curr_small,
            cv2.COLOR_BGR2RGB,
        )

        # ==========================================================
        # NUMPY → TORCH
        # ==========================================================

        prev_tensor = (
            torch.from_numpy(
                prev_rgb
            )
            .permute(
                2,
                0,
                1,
            )
            .float()
            / 255.0
        )

        curr_tensor = (
            torch.from_numpy(
                curr_rgb
            )
            .permute(
                2,
                0,
                1,
            )
            .float()
            / 255.0
        )

        # ==========================================================
        # BATCH
        # ==========================================================

        prev_tensor = (
            prev_tensor
            .unsqueeze(0)
            .to(self.device)
        )

        curr_tensor = (
            curr_tensor
            .unsqueeze(0)
            .to(self.device)
        )

        # ==========================================================
        # RAFT INFERENCE
        # ==========================================================

        inference_start = (
            time.perf_counter()
        )

        with torch.inference_mode():

            flow_predictions = (
                self.model(
                    prev_tensor,
                    curr_tensor,
                )
            )

        inference_time = (
            time.perf_counter()
            -
            inference_start
        )

        # ==========================================================
        # FINAL FLOW
        # ==========================================================

        flow = (
            flow_predictions[-1]
            .detach()
            .cpu()
            .numpy()
            [0]
            .transpose(
                1,
                2,
                0,
            )
        )

        # ==========================================================
        # FLOW COMPONENTS
        # ==========================================================

        flow_x = flow[:, :, 0]

        flow_y = flow[:, :, 1]

        magnitude = np.sqrt(
            flow_x ** 2
            +
            flow_y ** 2
        )

        # ==========================================================
        # ROBUST STATISTICS
        # ==========================================================

        valid = np.isfinite(
            magnitude
        )

        valid_magnitude = (
            magnitude[valid]
        )

        if (
            valid_magnitude.size
            == 0
        ):

            return

        # Remove pathological values
        # from statistics.

        valid_magnitude = (
            valid_magnitude[
                valid_magnitude
                <
                self.MAX_FLOW
            ]
        )

        if (
            valid_magnitude.size
            == 0
        ):

            return

        mean_flow = float(
            np.mean(
                valid_magnitude
            )
        )

        median_flow = float(
            np.median(
                valid_magnitude
            )
        )

        p90 = float(
            np.percentile(
                valid_magnitude,
                90,
            )
        )

        p95 = float(
            np.percentile(
                valid_magnitude,
                95,
            )
        )

        p99 = float(
            np.percentile(
                valid_magnitude,
                99,
            )
        )

        max_flow = float(
            np.max(
                valid_magnitude
            )
        )

        over_200 = float(
            np.mean(
                valid_magnitude
                >
                200.0
            )
            * 100.0
        )

        # ==========================================================
        # PUBLISH VISUALIZATION
        # ==========================================================

        self.publish_flow_visualization(
            flow,
            magnitude,
            stamp,
        )

        self.processed += 1

        fps = (
            1.0
            /
            inference_time
            if inference_time > 0
            else 0.0
        )

        # ==========================================================
        # LOG
        # ==========================================================

        if self.processed % 5 == 0:

            self.get_logger().info(
                "RAFT flow | "
                f"processed={self.processed} | "
                f"received={self.received} | "
                f"dropped={self.dropped} | "
                f"dt={dt:.3f}s | "
                f"inference={inference_time:.2f}s | "
                f"FPS={fps:.2f} | "
                f"mean={mean_flow:.2f}px | "
                f"median={median_flow:.2f}px | "
                f"P90={p90:.2f}px | "
                f"P95={p95:.2f}px | "
                f"P99={p99:.2f}px | "
                f"max={max_flow:.2f}px | "
                f">200px={over_200:.1f}%"
            )

    # ==============================================================
    # VISUALIZATION
    # ==============================================================

    def publish_flow_visualization(
        self,
        flow,
        magnitude,
        stamp,
    ):

        # ==========================================================
        # HSV FLOW VISUALIZATION
        # ==========================================================

        flow_x = flow[:, :, 0]

        flow_y = flow[:, :, 1]

        angle = np.arctan2(
            flow_y,
            flow_x,
        )

        angle = (
            angle + np.pi
        ) / (
            2.0 * np.pi
        )

        mag_vis = np.clip(
            magnitude / 100.0,
            0.0,
            1.0,
        )

        hsv = np.zeros(
            (
                self.INPUT_HEIGHT,
                self.INPUT_WIDTH,
                3,
            ),
            dtype=np.uint8,
        )

        hsv[:, :, 0] = (
            angle * 179.0
        ).astype(
            np.uint8
        )

        hsv[:, :, 1] = 255

        hsv[:, :, 2] = (
            mag_vis * 255.0
        ).astype(
            np.uint8
        )

        flow_bgr = cv2.cvtColor(
            hsv,
            cv2.COLOR_HSV2BGR,
        )

        # ==========================================================
        # MAGNITUDE IMAGE
        # ==========================================================

        magnitude_vis = np.clip(
            magnitude / 100.0
            * 255.0,
            0,
            255,
        ).astype(
            np.uint8
        )

        # ==========================================================
        # MESSAGE
        # ==========================================================

        flow_msg = (
            self.bridge.cv2_to_imgmsg(
                flow_bgr,
                encoding="bgr8",
            )
        )

        flow_msg.header.stamp = (
            stamp
        )

        flow_msg.header.frame_id = (
            "camera_front"
        )

        magnitude_msg = (
            self.bridge.cv2_to_imgmsg(
                magnitude_vis,
                encoding="mono8",
            )
        )

        magnitude_msg.header.stamp = (
            stamp
        )

        magnitude_msg.header.frame_id = (
            "camera_front"
        )

        # ==========================================================
        # PUBLISH
        # ==========================================================

        self.flow_overlay_pub.publish(
            flow_msg
        )

        self.flow_magnitude_pub.publish(
            magnitude_msg
        )

    # ==============================================================
    # SHUTDOWN
    # ==============================================================

    def stop_worker(self):

        with self.worker_condition:

            self.worker_running = False

            self.worker_condition.notify_all()

        if (
            self.worker_thread.is_alive()
        ):

            self.worker_thread.join(
                timeout=2.0
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

        node.stop_worker()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()