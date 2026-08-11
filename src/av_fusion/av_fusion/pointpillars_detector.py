import os
import pickle
import subprocess
import sys
import threading
import queue
import time

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from vision_msgs.msg import (
    Detection3DArray,
    Detection3D,
    BoundingBox3D,
)

from visualization_msgs.msg import (
    Marker,
    MarkerArray,
)


class PointPillarsDetector(Node):

    def __init__(self):

        super().__init__(
            "pointpillars_detector"
        )

        # ======================================================
        # CONFIG
        # ======================================================

        self.worker_python = (
            "/home/adarsh/miniconda3/envs/"
            "pointpillars/bin/python"
        )

        self.worker_script = (
            "/home/adarsh/ml/OpenPCDet/"
            "pointpillars_worker.py"
        )

        self.input_topic = (
            "/lidar/points"
        )

        self.detection_topic = (
            "/detections/boxes_3d"
        )

        self.marker_topic = (
            "/detections/markers"
        )

        # ======================================================
        # SUBSCRIBER
        # ======================================================

        self.subscription = (
            self.create_subscription(
                PointCloud2,
                self.input_topic,
                self.lidar_callback,
                10
            )
        )

        # ======================================================
        # PUBLISHERS
        # ======================================================

        self.detection_pub = (
            self.create_publisher(
                Detection3DArray,
                self.detection_topic,
                10
            )
        )

        self.marker_pub = (
            self.create_publisher(
                MarkerArray,
                self.marker_topic,
                10
            )
        )

        # ======================================================
        # WORKER
        # ======================================================

        self.get_logger().info(
            "Starting PointPillars Python 3.8 worker..."
        )

        self.worker = subprocess.Popen(
            [
                self.worker_python,
                self.worker_script,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0
        )

        # ======================================================
        # WORKER STATE
        # ======================================================

        self.worker_lock = threading.Lock()

        self.inference_running = False

        self.frame_counter = 0

        # ======================================================
        # STARTUP
        # ======================================================

        self.get_logger().info(
            "=============================================="
        )

        self.get_logger().info(
            "PointPillars ROS2 Detector"
        )

        self.get_logger().info(
            f"Input: {self.input_topic}"
        )

        self.get_logger().info(
            f"Output: {self.detection_topic}"
        )

        self.get_logger().info(
            f"Markers: {self.marker_topic}"
        )

        self.get_logger().info(
            "CPU PointPillars worker started."
        )

        self.get_logger().info(
            "=============================================="
        )

    # ==========================================================
    # POINTCLOUD → NUMPY
    # ==========================================================

    def pointcloud_to_numpy(
        self,
        msg
    ):

        # ------------------------------------------------------
        # Read exactly the fields published by our NuScenes
        # player.
        # ------------------------------------------------------

        points = []

        for p in point_cloud2.read_points(
            msg,
            field_names=(
                "x",
                "y",
                "z",
                "intensity",
                "timestamp"
            ),
            skip_nans=True
        ):

            points.append(
                [
                    p[0],
                    p[1],
                    p[2],
                    p[3],
                    p[4]
                ]
            )

        if len(points) == 0:

            return np.empty(
                (0, 5),
                dtype=np.float32
            )

        points = np.asarray(
            points,
            dtype=np.float32
        )

        return points

    # ==========================================================
    # CALLBACK
    # ==========================================================

    def lidar_callback(
        self,
        msg
    ):

        # ------------------------------------------------------
        # Don't queue another frame while CPU inference is
        # running.
        #
        # Your current inference is ~28 seconds for only
        # 1000 random points, so accumulating frames would
        # create an enormous backlog.
        # ------------------------------------------------------

        if self.inference_running:

            self.get_logger().warn(
                "PointPillars inference still running; "
                "dropping incoming LiDAR frame."
            )

            return

        # ------------------------------------------------------
        # Convert PointCloud2
        # ------------------------------------------------------

        points = self.pointcloud_to_numpy(
            msg
        )

        if points.shape[0] == 0:

            self.get_logger().warn(
                "Received empty LiDAR cloud."
            )

            return

        self.frame_counter += 1

        frame_id = (
            f"ros_frame_{self.frame_counter}"
        )

        self.get_logger().info(
            f"[{frame_id}] "
            f"Received {points.shape[0]} points."
        )

        # ------------------------------------------------------
        # Run inference in background thread.
        #
        # This prevents the ROS executor from being blocked
        # for ~28 seconds.
        # ------------------------------------------------------

        self.inference_running = True

        thread = threading.Thread(
            target=self.run_worker,
            args=(
                points,
                frame_id,
                msg.header
            ),
            daemon=True
        )

        thread.start()

    # ==========================================================
    # SEND TO PYTHON 3.8 WORKER
    # ==========================================================

    def run_worker(
        self,
        points,
        frame_id,
        header
    ):

        start = time.time()

        try:

            request = {
                "frame_id": frame_id,
                "points": points
            }

            # --------------------------------------------------
            # Only one thread can communicate with the worker.
            # --------------------------------------------------

            with self.worker_lock:

                pickle.dump(
                    request,
                    self.worker.stdin
                )

                self.worker.stdin.flush()

                # ----------------------------------------------
                # Wait for inference.
                # ----------------------------------------------

                result = pickle.load(
                    self.worker.stdout
                )

            # --------------------------------------------------
            # Check worker result
            # --------------------------------------------------

            if not result.get(
                "ok",
                False
            ):

                self.get_logger().error(
                    f"[{frame_id}] "
                    f"PointPillars error: "
                    f"{result.get('error')}"
                )

                return

            # --------------------------------------------------
            # Extract predictions
            # --------------------------------------------------

            pred_boxes = np.asarray(
                result["pred_boxes"],
                dtype=np.float32
            )

            pred_scores = np.asarray(
                result["pred_scores"],
                dtype=np.float32
            )

            pred_labels = np.asarray(
                result["pred_labels"],
                dtype=np.int64
            )

            elapsed = (
                time.time() - start
            )

            self.get_logger().info(
                f"[{frame_id}] "
                f"detections={len(pred_scores)} "
                f"inference={elapsed:.2f}s"
            )

            # --------------------------------------------------
            # Publish ROS detections
            # --------------------------------------------------

            self.publish_detections(
                pred_boxes,
                pred_scores,
                pred_labels,
                header
            )

            # --------------------------------------------------
            # Publish RViz markers
            # --------------------------------------------------

            self.publish_markers(
                pred_boxes,
                pred_scores,
                pred_labels,
                header
            )

        except Exception as e:

            self.get_logger().error(
                f"[{frame_id}] "
                f"Worker communication failed: {e}"
            )

        finally:

            self.inference_running = False

    # ==========================================================
    # PUBLISH Detection3DArray
    # ==========================================================

    def publish_detections(
        self,
        boxes,
        scores,
        labels,
        header
    ):

        msg = Detection3DArray()

        msg.header = header

        # ------------------------------------------------------
        # One PointPillars prediction:
        #
        # [x, y, z, dx, dy, dz, heading]
        #
        # Some models can output:
        #
        # [x, y, z, dx, dy, dz, heading, vx, vy]
        #
        # so we handle both.
        # ------------------------------------------------------

        for i in range(
            len(scores)
        ):

            box = boxes[i]

            if len(box) < 7:

                continue

            detection = Detection3D()

            detection.header = header

            bbox = BoundingBox3D()

            # --------------------------------------------------
            # Center
            # --------------------------------------------------

            bbox.center.position.x = (
                float(box[0])
            )

            bbox.center.position.y = (
                float(box[1])
            )

            bbox.center.position.z = (
                float(box[2])
            )

            # --------------------------------------------------
            # Heading
            # --------------------------------------------------

            heading = float(
                box[6]
            )

            bbox.center.orientation.z = (
                np.sin(
                    heading / 2.0
                )
            )

            bbox.center.orientation.w = (
                np.cos(
                    heading / 2.0
                )
            )

            # --------------------------------------------------
            # Dimensions
            # --------------------------------------------------

            bbox.size.x = float(
                box[3]
            )

            bbox.size.y = float(
                box[4]
            )

            bbox.size.z = float(
                box[5]
            )

            detection.bbox = bbox

            msg.detections.append(
                detection
            )

        self.detection_pub.publish(
            msg
        )

    # ==========================================================
    # RVIZ MARKERS
    # ==========================================================

    def publish_markers(
        self,
        boxes,
        scores,
        labels,
        header
    ):

        marker_array = MarkerArray()

        # ------------------------------------------------------
        # Delete previous markers
        # ------------------------------------------------------

        delete = Marker()

        delete.header = header

        delete.ns = (
            "pointpillars_detections"
        )

        delete.action = (
            Marker.DELETEALL
        )

        marker_array.markers.append(
            delete
        )

        # ------------------------------------------------------
        # Add predictions
        # ------------------------------------------------------

        for i in range(
            len(scores)
        ):

            box = boxes[i]

            if len(box) < 7:

                continue

            marker = Marker()

            marker.header = header

            marker.ns = (
                "pointpillars_detections"
            )

            marker.id = i

            marker.type = (
                Marker.CUBE
            )

            marker.action = (
                Marker.ADD
            )

            # --------------------------------------------------
            # Position
            # --------------------------------------------------

            marker.pose.position.x = (
                float(box[0])
            )

            marker.pose.position.y = (
                float(box[1])
            )

            marker.pose.position.z = (
                float(box[2])
            )

            # --------------------------------------------------
            # Orientation
            # --------------------------------------------------

            heading = float(
                box[6]
            )

            marker.pose.orientation.z = (
                np.sin(
                    heading / 2.0
                )
            )

            marker.pose.orientation.w = (
                np.cos(
                    heading / 2.0
                )
            )

            # --------------------------------------------------
            # Size
            # --------------------------------------------------

            marker.scale.x = float(
                box[3]
            )

            marker.scale.y = float(
                box[4]
            )

            marker.scale.z = float(
                box[5]
            )

            # --------------------------------------------------
            # Color
            # --------------------------------------------------

            marker.color.r = 0.0
            marker.color.g = 0.8
            marker.color.b = 1.0
            marker.color.a = 0.55

            # --------------------------------------------------
            # Lifetime
            # --------------------------------------------------

            marker.lifetime.sec = 0
            marker.lifetime.nanosec = (
                800000000
            )

            marker_array.markers.append(
                marker
            )

        self.marker_pub.publish(
            marker_array
        )

    # ==========================================================
    # SHUTDOWN
    # ==========================================================

    def destroy_node(
        self
    ):

        self.get_logger().info(
            "Stopping PointPillars worker..."
        )

        try:

            if self.worker.poll() is None:

                self.worker.terminate()

                self.worker.wait(
                    timeout=5
                )

        except Exception:

            try:
                self.worker.kill()
            except Exception:
                pass

        super().destroy_node()


# ==============================================================
# MAIN
# ==============================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = PointPillarsDetector()

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