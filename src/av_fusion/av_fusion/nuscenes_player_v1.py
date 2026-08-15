#!/usr/bin/env python3

import os
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

from sensor_msgs.msg import (
    Image,
    PointCloud2,
    PointField,
)

from visualization_msgs.msg import (
    Marker,
    MarkerArray,
)

from vision_msgs.msg import (
    Detection3DArray,
    Detection3D,
    BoundingBox3D,
)

from cv_bridge import CvBridge

import tf2_ros

from geometry_msgs.msg import TransformStamped

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

from sensor_msgs_py import point_cloud2


class NuScenesAVPlayer(Node):

    def __init__(self):

        super().__init__("nuscenes_av_player")

        # ==========================================================
        # CONFIGURATION
        # ==========================================================

        self.nusc_root = os.environ.get(
            "NUSCENES_DIR",
            "/home/adarsh/av_perception/data/nuscenes"
        )

        self.version = "v1.0-mini"

        self.cam = "CAM_FRONT"

        self.lidar = "LIDAR_TOP"

        # Number of LiDAR sweeps.
        #
        # Keep this at 10 because this matches the configuration
        # we have been using for the LiDAR pipeline.
        #
        self.num_lidar_sweeps = 10

        # ==========================================================
        # PLAYBACK
        # ==========================================================

        # NuScenes mini keyframes are approximately 2 Hz.
        #
        # IMPORTANT:
        #
        # This timer ONLY handles:
        #
        #   - loading camera image
        #   - publishing camera image
        #   - advancing the sample
        #
        # Heavy LiDAR processing happens in a worker thread.
        #
        self.playback_period = 0.5

        # ==========================================================
        # QOS
        # ==========================================================

        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.bridge = CvBridge()

        # ==========================================================
        # PUBLISHERS
        # ==========================================================

        # ----------------------------------------------------------
        # Camera
        # ----------------------------------------------------------

        self.image_pub = self.create_publisher(
            Image,
            "/camera/front/image",
            qos,
        )

        # ----------------------------------------------------------
        # LiDAR
        # ----------------------------------------------------------

        self.lidar_pub = self.create_publisher(
            PointCloud2,
            "/lidar/points",
            qos,
        )

        # ----------------------------------------------------------
        # Structured GT
        # ----------------------------------------------------------

        self.gt_pub = self.create_publisher(
            Detection3DArray,
            "/detections/boxes_3d",
            qos,
        )

        # ----------------------------------------------------------
        # GT visualization
        # ----------------------------------------------------------

        self.gt_vis_pub = self.create_publisher(
            MarkerArray,
            "/gt/visualization_markers",
            qos,
        )

        # ==========================================================
        # TF
        # ==========================================================

        self.tf_broadcaster = (
            tf2_ros.TransformBroadcaster(self)
        )

        # ==========================================================
        # NUSCENES
        # ==========================================================

        self.get_logger().info(
            f"Loading NuScenes dataset from: "
            f"{self.nusc_root}"
        )

        self.nusc = NuScenes(
            version=self.version,
            dataroot=self.nusc_root,
            verbose=False,
        )

        # ==========================================================
        # SCENE
        # ==========================================================

        self.scene = self.nusc.scene[0]

        self.first_sample_token = (
            self.scene["first_sample_token"]
        )

        self.sample_token = (
            self.first_sample_token
        )

        # ==========================================================
        # WORKER STATE
        # ==========================================================

        # The camera thread produces samples.
        #
        # The LiDAR worker consumes samples.
        #
        # We intentionally keep only ONE pending sample.
        #
        # This prevents:
        #
        # camera:  sample 1,2,3,4,5,6...
        #
        # worker:  stuck processing sample 1
        #
        # from creating an infinite backlog.
        #
        self.pending_sample = None

        self.pending_lock = threading.Lock()

        self.worker_condition = (
            threading.Condition(
                self.pending_lock
            )
        )

        self.worker_running = True

        self.worker_thread = threading.Thread(
            target=self.heavy_processing_worker,
            daemon=True,
        )

        self.worker_thread.start()

        # ==========================================================
        # TIMER
        # ==========================================================

        self.timer = self.create_timer(
            self.playback_period,
            self.camera_timer_callback,
        )

        # ==========================================================
        # STATISTICS
        # ==========================================================

        self.camera_frames = 0

        self.camera_errors = 0

        self.worker_processed = 0

        self.worker_dropped = 0

        self.last_camera_wall_time = None

        self.last_camera_dataset_timestamp = None

        # ==========================================================
        # STARTUP LOG
        # ==========================================================

        self.get_logger().info(
            "=============================================="
        )

        self.get_logger().info(
            "NuScenes AV Player Started"
        )

        self.get_logger().info(
            f"Dataset: {self.version}"
        )

        self.get_logger().info(
            f"LiDAR sweeps: {self.num_lidar_sweeps}"
        )

        self.get_logger().info(
            f"Camera playback period: "
            f"{self.playback_period:.2f}s"
        )

        self.get_logger().info(
            "Camera playback is decoupled "
            "from heavy LiDAR processing."
        )

        self.get_logger().info(
            "Using NuScenes sample timestamps."
        )

        self.get_logger().info(
            "Latest-sample-only worker queue enabled."
        )

        self.get_logger().info(
            "Publishing:"
        )

        self.get_logger().info(
            "  /camera/front/image"
        )

        self.get_logger().info(
            "  /lidar/points"
        )

        self.get_logger().info(
            "  /detections/boxes_3d"
        )

        self.get_logger().info(
            "  /gt/visualization_markers"
        )

        self.get_logger().info(
            "=============================================="
        )

    # ==============================================================
    # NUSCENES TIMESTAMP
    # ==============================================================

    def sample_timestamp_to_ros_time(
        self,
        sample,
    ):
        """
        Convert NuScenes sample timestamp from microseconds
        to ROS builtin_interfaces/Time.

        NuScenes:
            timestamp = microseconds

        ROS:
            seconds + nanoseconds
        """

        timestamp_us = int(
            sample["timestamp"]
        )

        sec = timestamp_us // 1_000_000

        nanosec = (
            timestamp_us
            % 1_000_000
        ) * 1000

        ros_time = self.get_clock().now().to_msg()

        ros_time.sec = int(sec)

        ros_time.nanosec = int(
            nanosec
        )

        return ros_time

    # ==============================================================
    # CAMERA TIMER
    # ==============================================================

    def camera_timer_callback(self):

        callback_start = time.perf_counter()

        # ==========================================================
        # LOOP
        # ==========================================================

        if self.sample_token == "":

            self.get_logger().info(
                "=============================================="
            )

            self.get_logger().info(
                "End of NuScenes scene reached."
            )

            self.get_logger().info(
                "Restarting scene from first sample."
            )

            self.get_logger().info(
                "=============================================="
            )

            self.sample_token = (
                self.first_sample_token
            )

        # ==========================================================
        # CURRENT SAMPLE
        # ==========================================================

        sample = self.nusc.get(
            "sample",
            self.sample_token,
        )

        # ==========================================================
        # DATASET TIMESTAMP
        # ==========================================================

        timestamp = (
            self.sample_timestamp_to_ros_time(
                sample
            )
        )

        # ==========================================================
        # CAMERA
        # ==========================================================

        cam_data = self.nusc.get(
            "sample_data",
            sample["data"][self.cam],
        )

        img_path = os.path.join(
            self.nusc_root,
            cam_data["filename"],
        )

        img = cv2.imread(
            img_path
        )

        if img is None:

            self.camera_errors += 1

            self.get_logger().error(
                f"Could not read image: "
                f"{img_path}"
            )

            # Still advance so that a bad frame
            # does not freeze the whole dataset.
            self.sample_token = (
                sample["next"]
            )

            return

        # ==========================================================
        # CREATE IMAGE MESSAGE
        # ==========================================================

        img_msg = (
            self.bridge.cv2_to_imgmsg(
                img,
                encoding="bgr8",
            )
        )

        # IMPORTANT:
        #
        # Use DATASET timestamp.
        #
        img_msg.header.stamp = timestamp

        img_msg.header.frame_id = (
            "camera_front"
        )

        # ==========================================================
        # PUBLISH CAMERA
        # ==========================================================

        self.image_pub.publish(
            img_msg
        )

        self.camera_frames += 1

        # ==========================================================
        # QUEUE SAMPLE FOR HEAVY WORK
        # ==========================================================

        self.queue_heavy_processing(
            sample,
            timestamp,
        )

        # ==========================================================
        # CAMERA TIMING LOG
        # ==========================================================

        wall_now = time.perf_counter()

        wall_dt = None

        if self.last_camera_wall_time is not None:

            wall_dt = (
                wall_now
                -
                self.last_camera_wall_time
            )

        dataset_dt = None

        if (
            self.last_camera_dataset_timestamp
            is not None
        ):

            dataset_dt = (
                (
                    sample["timestamp"]
                    -
                    self.last_camera_dataset_timestamp
                )
                /
                1_000_000.0
            )

        self.last_camera_wall_time = (
            wall_now
        )

        self.last_camera_dataset_timestamp = (
            sample["timestamp"]
        )

        callback_time = (
            time.perf_counter()
            -
            callback_start
        )

        # Log every 10 camera frames.

        if self.camera_frames % 10 == 0:

            wall_text = (
                f"{wall_dt:.3f}s"
                if wall_dt is not None
                else "N/A"
            )

            dataset_text = (
                f"{dataset_dt:.3f}s"
                if dataset_dt is not None
                else "N/A"
            )

            self.get_logger().info(
                "Camera playback | "
                f"frame={self.camera_frames} | "
                f"callback={callback_time:.3f}s | "
                f"wall_dt={wall_text} | "
                f"dataset_dt={dataset_text} | "
                f"worker={self.worker_processed} | "
                f"dropped={self.worker_dropped}"
            )

        # ==========================================================
        # ADVANCE SAMPLE
        # ==========================================================

        self.sample_token = (
            sample["next"]
        )

    # ==============================================================
    # QUEUE HEAVY PROCESSING
    # ==============================================================

    def queue_heavy_processing(
        self,
        sample,
        timestamp,
    ):

        with self.worker_condition:

            # If a previous sample is still waiting,
            # replace it with the newest one.
            #
            # We care about the latest sensor state,
            # not processing every stale frame.

            if self.pending_sample is not None:

                self.worker_dropped += 1

            self.pending_sample = (
                sample,
                timestamp,
            )

            self.worker_condition.notify()

    # ==============================================================
    # HEAVY PROCESSING WORKER
    # ==============================================================

    def heavy_processing_worker(self):

        self.get_logger().info(
            "Heavy-processing worker started."
        )

        while self.worker_running:

            # ======================================================
            # WAIT FOR SAMPLE
            # ======================================================

            with self.worker_condition:

                while (
                    self.pending_sample is None
                    and
                    self.worker_running
                ):

                    self.worker_condition.wait(
                        timeout=0.5
                    )

                if not self.worker_running:

                    break

                work = (
                    self.pending_sample
                )

                self.pending_sample = None

            if work is None:

                continue

            sample, timestamp = work

            # ======================================================
            # PROCESS
            # ======================================================

            worker_start = time.perf_counter()

            try:

                # --------------------------------------------------
                # LIDAR
                # --------------------------------------------------

                self.process_lidar(
                    sample,
                    timestamp,
                )

                # --------------------------------------------------
                # TF
                # --------------------------------------------------

                self.publish_tf(
                    sample,
                    timestamp,
                )

                # --------------------------------------------------
                # GT
                # --------------------------------------------------

                self.publish_gt_boxes(
                    sample,
                    timestamp,
                )

                self.worker_processed += 1

            except Exception as e:

                self.get_logger().error(
                    "Heavy processing failed: "
                    f"{e}"
                )

            worker_time = (
                time.perf_counter()
                -
                worker_start
            )

            if self.worker_processed % 5 == 0:

                self.get_logger().info(
                    "Heavy processing | "
                    f"processed="
                    f"{self.worker_processed} | "
                    f"time={worker_time:.2f}s | "
                    f"dropped="
                    f"{self.worker_dropped}"
                )

        self.get_logger().info(
            "Heavy-processing worker stopped."
        )

    # ==============================================================
    # LIDAR PROCESSING
    # ==============================================================

    def process_lidar(
        self,
        sample,
        timestamp,
    ):

        # ==========================================================
        # LOAD MULTI-SWEEP LIDAR
        # ==========================================================

        pc, times = (
            LidarPointCloud.from_file_multisweep(
                self.nusc,
                sample,
                self.lidar,
                self.lidar,
                nsweeps=self.num_lidar_sweeps,
            )
        )

        # ==========================================================
        # CHECK POINTS
        # ==========================================================

        if pc.points.shape[1] == 0:

            self.get_logger().warn(
                "No LiDAR points found."
            )

            return

        # ==========================================================
        # POINT FORMAT
        #
        # pc.points:
        #
        # [x]
        # [y]
        # [z]
        # [intensity]
        #
        # times:
        #
        # time lag
        # ==========================================================

        points = np.vstack(
            [
                pc.points,
                times,
            ]
        ).T.astype(
            np.float32
        )

        # ==========================================================
        # REMOVE INVALID VALUES
        # ==========================================================

        finite_mask = np.all(
            np.isfinite(points),
            axis=1,
        )

        points = points[
            finite_mask
        ]

        # ==========================================================
        # HEADER
        # ==========================================================

        lidar_header = (
            self.create_header(
                timestamp,
                "lidar_top",
            )
        )

        # ==========================================================
        # POINTCLOUD
        # ==========================================================

        cloud_msg = (
            self.create_lidar_cloud_msg(
                points,
                lidar_header,
            )
        )

        # ==========================================================
        # PUBLISH
        # ==========================================================

        self.lidar_pub.publish(
            cloud_msg
        )

    # ==============================================================
    # CREATE HEADER
    # ==============================================================

    def create_header(
        self,
        timestamp,
        frame_id,
    ):

        # We can construct a Header without needing
        # an intermediate image message.

        from std_msgs.msg import Header

        header = Header()

        header.stamp = timestamp

        header.frame_id = frame_id

        return header

    # ==============================================================
    # TF
    # ==============================================================

    def publish_tf(
        self,
        sample,
        timestamp,
    ):

        # ==========================================================
        # CAMERA SAMPLE DATA
        # ==========================================================

        cam_data = self.nusc.get(
            "sample_data",
            sample["data"][self.cam],
        )

        ego_pose = self.nusc.get(
            "ego_pose",
            cam_data["ego_pose_token"],
        )

        # ==========================================================
        # MAP → BASE_LINK
        # ==========================================================

        t = TransformStamped()

        t.header.stamp = timestamp

        t.header.frame_id = (
            "map"
        )

        t.child_frame_id = (
            "base_link"
        )

        t.transform.translation.x = float(
            ego_pose["translation"][0]
        )

        t.transform.translation.y = float(
            ego_pose["translation"][1]
        )

        t.transform.translation.z = float(
            ego_pose["translation"][2]
        )

        q = ego_pose["rotation"]

        t.transform.rotation.x = float(
            q[1]
        )

        t.transform.rotation.y = float(
            q[2]
        )

        t.transform.rotation.z = float(
            q[3]
        )

        t.transform.rotation.w = float(
            q[0]
        )

        self.tf_broadcaster.sendTransform(
            t
        )

        # ==========================================================
        # BASE_LINK → LIDAR_TOP
        # ==========================================================

        lidar_data = self.nusc.get(
            "sample_data",
            sample["data"][self.lidar],
        )

        lidar_cs = self.nusc.get(
            "calibrated_sensor",
            lidar_data[
                "calibrated_sensor_token"
            ],
        )

        t2 = TransformStamped()

        t2.header.stamp = timestamp

        t2.header.frame_id = (
            "base_link"
        )

        t2.child_frame_id = (
            "lidar_top"
        )

        t2.transform.translation.x = float(
            lidar_cs["translation"][0]
        )

        t2.transform.translation.y = float(
            lidar_cs["translation"][1]
        )

        t2.transform.translation.z = float(
            lidar_cs["translation"][2]
        )

        q = lidar_cs["rotation"]

        t2.transform.rotation.x = float(
            q[1]
        )

        t2.transform.rotation.y = float(
            q[2]
        )

        t2.transform.rotation.z = float(
            q[3]
        )

        t2.transform.rotation.w = float(
            q[0]
        )

        self.tf_broadcaster.sendTransform(
            t2
        )

    # ==============================================================
    # GROUND TRUTH BOXES
    # ==============================================================

    def publish_gt_boxes(
        self,
        sample,
        timestamp,
    ):

        # ==========================================================
        # DETECTION ARRAY
        # ==========================================================

        detections_msg = (
            Detection3DArray()
        )

        detections_msg.header.frame_id = (
            "map"
        )

        detections_msg.header.stamp = (
            timestamp
        )

        # ==========================================================
        # MARKERS
        # ==========================================================

        marker_array = MarkerArray()

        # ----------------------------------------------------------
        # Remove previous markers
        # ----------------------------------------------------------

        delete_all = Marker()

        delete_all.header.frame_id = (
            "map"
        )

        delete_all.header.stamp = (
            timestamp
        )

        delete_all.ns = (
            "gt_boxes"
        )

        delete_all.action = (
            Marker.DELETEALL
        )

        marker_array.markers.append(
            delete_all
        )

        # ==========================================================
        # ANNOTATIONS
        # ==========================================================

        for i, ann_token in enumerate(
            sample["anns"]
        ):

            ann = self.nusc.get(
                "sample_annotation",
                ann_token,
            )

            # ======================================================
            # Detection3D
            # ======================================================

            detection = Detection3D()

            detection.header = (
                detections_msg.header
            )

            bbox = BoundingBox3D()

            # ------------------------------------------------------
            # POSITION
            # ------------------------------------------------------

            bbox.center.position.x = float(
                ann["translation"][0]
            )

            bbox.center.position.y = float(
                ann["translation"][1]
            )

            bbox.center.position.z = float(
                ann["translation"][2]
            )

            # ------------------------------------------------------
            # ORIENTATION
            # ------------------------------------------------------

            q = ann["rotation"]

            bbox.center.orientation.x = float(
                q[1]
            )

            bbox.center.orientation.y = float(
                q[2]
            )

            bbox.center.orientation.z = float(
                q[3]
            )

            bbox.center.orientation.w = float(
                q[0]
            )

            # ------------------------------------------------------
            # SIZE
            #
            # NuScenes:
            #
            # [width, length, height]
            #
            # ROS:
            #
            # x = length
            # y = width
            # z = height
            # ------------------------------------------------------

            bbox.size.x = float(
                ann["size"][1]
            )

            bbox.size.y = float(
                ann["size"][0]
            )

            bbox.size.z = float(
                ann["size"][2]
            )

            detection.bbox = bbox

            detections_msg.detections.append(
                detection
            )

            # ======================================================
            # RVIZ MARKER
            # ======================================================

            marker = Marker()

            marker.header = (
                detections_msg.header
            )

            marker.ns = (
                "gt_boxes"
            )

            marker.id = i

            marker.type = (
                Marker.CUBE
            )

            marker.action = (
                Marker.ADD
            )

            marker.pose = (
                bbox.center
            )

            marker.scale = (
                bbox.size
            )

            # ------------------------------------------------------
            # Green GT
            # ------------------------------------------------------

            marker.color.r = 0.0

            marker.color.g = 1.0

            marker.color.b = 0.0

            marker.color.a = 0.5

            # ------------------------------------------------------
            # Lifetime
            # ------------------------------------------------------

            marker.lifetime.sec = 0

            marker.lifetime.nanosec = (
                800000000
            )

            marker_array.markers.append(
                marker
            )

        # ==========================================================
        # PUBLISH
        # ==========================================================

        self.gt_pub.publish(
            detections_msg
        )

        self.gt_vis_pub.publish(
            marker_array
        )

    # ==============================================================
    # CREATE POINTCLOUD2
    # ==============================================================

    def create_lidar_cloud_msg(
        self,
        points,
        header,
    ):
        """
        Create PointCloud2:

            x
            y
            z
            intensity
            timestamp

        points shape:

            (N, 5)
        """

        # ==========================================================
        # POINT FIELDS
        # ==========================================================

        fields = [

            PointField(
                name="x",
                offset=0,
                datatype=PointField.FLOAT32,
                count=1,
            ),

            PointField(
                name="y",
                offset=4,
                datatype=PointField.FLOAT32,
                count=1,
            ),

            PointField(
                name="z",
                offset=8,
                datatype=PointField.FLOAT32,
                count=1,
            ),

            PointField(
                name="intensity",
                offset=12,
                datatype=PointField.FLOAT32,
                count=1,
            ),

            PointField(
                name="timestamp",
                offset=16,
                datatype=PointField.FLOAT32,
                count=1,
            ),
        ]

        # ==========================================================
        # CREATE
        # ==========================================================

        cloud_msg = point_cloud2.create_cloud(
            header,
            fields,
            points,
        )

        return cloud_msg

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

    node = NuScenesAVPlayer()

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


# ==================================================================
# ENTRY POINT
# ==================================================================

if __name__ == "__main__":

    main()