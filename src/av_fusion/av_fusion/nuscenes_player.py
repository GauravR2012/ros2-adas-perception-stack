import os

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

from geometry_msgs.msg import TransformStamped

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

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

from sensor_msgs_py import point_cloud2


class NuScenesAVPlayer(Node):

    def __init__(self):

        super().__init__("nuscenes_av_player")

        # ======================================================
        # CONFIG
        # ======================================================

        self.nusc_root = os.environ.get(
            "NUSCENES_DIR",
            "/home/adarsh/av_perception/data/nuscenes"
        )

        self.version = "v1.0-mini"

        self.cam = "CAM_FRONT"

        self.lidar = "LIDAR_TOP"

        # Number of LiDAR sweeps.
        #
        # This matches:
        #
        # MAX_SWEEPS: 10
        #
        # in your OpenPCDet NuScenes configuration.
        self.num_lidar_sweeps = 10

        # ======================================================
        # QoS
        # ======================================================

        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        self.bridge = CvBridge()

        # ======================================================
        # PUBLISHERS
        # ======================================================

        # ------------------------------------------------------
        # Camera
        # ------------------------------------------------------

        self.image_pub = self.create_publisher(
            Image,
            "/camera/front/image",
            qos
        )

        # ------------------------------------------------------
        # LiDAR
        #
        # This will now publish:
        #
        # x
        # y
        # z
        # intensity
        # timestamp
        #
        # instead of only x,y,z.
        # ------------------------------------------------------

        self.lidar_pub = self.create_publisher(
            PointCloud2,
            "/lidar/points",
            qos
        )

        # ------------------------------------------------------
        # Structured GT
        # ------------------------------------------------------

        self.gt_pub = self.create_publisher(
            Detection3DArray,
            "/detections/boxes_3d",
            qos
        )

        # ------------------------------------------------------
        # GT visualization
        # ------------------------------------------------------

        self.gt_vis_pub = self.create_publisher(
            MarkerArray,
            "/gt/visualization_markers",
            qos
        )

        # ======================================================
        # TF
        # ======================================================

        self.tf_broadcaster = (
            tf2_ros.TransformBroadcaster(self)
        )

        # ======================================================
        # NUSCENES
        # ======================================================

        self.get_logger().info(
            f"Loading NuScenes dataset from: "
            f"{self.nusc_root}"
        )

        self.nusc = NuScenes(
            version=self.version,
            dataroot=self.nusc_root,
            verbose=False
        )

        # ======================================================
        # SCENE
        # ======================================================

        self.scene = self.nusc.scene[0]

        self.first_sample_token = (
            self.scene["first_sample_token"]
        )

        self.sample_token = (
            self.first_sample_token
        )

        # ======================================================
        # TIMER
        #
        # 0.5 sec = 2 Hz playback
        #
        # Keep this low initially because your CPU
        # PointPillars inference will be relatively slow.
        # ======================================================

        self.timer = self.create_timer(
            0.5,
            self.timer_callback
        )

        # ======================================================
        # STARTUP LOG
        # ======================================================

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

    # ==========================================================
    # TF
    # ==========================================================

    def publish_tf(
        self,
        sample,
        timestamp
    ):

        # ======================================================
        # CAMERA SAMPLE DATA
        # ======================================================

        cam_data = self.nusc.get(
            "sample_data",
            sample["data"][self.cam]
        )

        ego_pose = self.nusc.get(
            "ego_pose",
            cam_data["ego_pose_token"]
        )

        # ======================================================
        # MAP → BASE_LINK
        # ======================================================

        t = TransformStamped()

        t.header.stamp = timestamp

        t.header.frame_id = "map"

        t.child_frame_id = "base_link"

        t.transform.translation.x = (
            float(ego_pose["translation"][0])
        )

        t.transform.translation.y = (
            float(ego_pose["translation"][1])
        )

        t.transform.translation.z = (
            float(ego_pose["translation"][2])
        )

        q = ego_pose["rotation"]

        t.transform.rotation.x = float(q[1])
        t.transform.rotation.y = float(q[2])
        t.transform.rotation.z = float(q[3])
        t.transform.rotation.w = float(q[0])

        self.tf_broadcaster.sendTransform(t)

        # ======================================================
        # BASE_LINK → LIDAR_TOP
        # ======================================================

        lidar_data = self.nusc.get(
            "sample_data",
            sample["data"][self.lidar]
        )

        lidar_cs = self.nusc.get(
            "calibrated_sensor",
            lidar_data["calibrated_sensor_token"]
        )

        t2 = TransformStamped()

        t2.header.stamp = timestamp

        t2.header.frame_id = "base_link"

        t2.child_frame_id = "lidar_top"

        t2.transform.translation.x = (
            float(lidar_cs["translation"][0])
        )

        t2.transform.translation.y = (
            float(lidar_cs["translation"][1])
        )

        t2.transform.translation.z = (
            float(lidar_cs["translation"][2])
        )

        q = lidar_cs["rotation"]

        t2.transform.rotation.x = float(q[1])
        t2.transform.rotation.y = float(q[2])
        t2.transform.rotation.z = float(q[3])
        t2.transform.rotation.w = float(q[0])

        self.tf_broadcaster.sendTransform(t2)

    # ==========================================================
    # GROUND TRUTH BOXES
    # ==========================================================

    def publish_gt_boxes(
        self,
        sample,
        timestamp
    ):

        # ======================================================
        # DETECTION ARRAY
        # ======================================================

        detections_msg = Detection3DArray()

        detections_msg.header.frame_id = "map"

        detections_msg.header.stamp = timestamp

        # ======================================================
        # MARKERS
        # ======================================================

        marker_array = MarkerArray()

        # ------------------------------------------------------
        # Remove previous GT markers.
        # ------------------------------------------------------

        delete_all = Marker()

        delete_all.header.frame_id = "map"

        delete_all.header.stamp = timestamp

        delete_all.ns = "gt_boxes"

        delete_all.action = Marker.DELETEALL

        marker_array.markers.append(
            delete_all
        )

        # ======================================================
        # ANNOTATIONS
        # ======================================================

        for i, ann_token in enumerate(
            sample["anns"]
        ):

            ann = self.nusc.get(
                "sample_annotation",
                ann_token
            )

            # ==================================================
            # Detection3D
            # ==================================================

            detection = Detection3D()

            detection.header = (
                detections_msg.header
            )

            bbox = BoundingBox3D()

            # --------------------------------------------------
            # Position
            # --------------------------------------------------

            bbox.center.position.x = float(
                ann["translation"][0]
            )

            bbox.center.position.y = float(
                ann["translation"][1]
            )

            bbox.center.position.z = float(
                ann["translation"][2]
            )

            # --------------------------------------------------
            # Orientation
            # --------------------------------------------------

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

            # --------------------------------------------------
            # NuScenes size is:
            #
            # [width, length, height]
            #
            # ROS BoundingBox3D uses:
            #
            # x = length
            # y = width
            # z = height
            # --------------------------------------------------

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

            # ==================================================
            # RVIZ GT MARKER
            # ==================================================

            marker = Marker()

            marker.header = (
                detections_msg.header
            )

            marker.ns = "gt_boxes"

            marker.id = i

            marker.type = Marker.CUBE

            marker.action = Marker.ADD

            marker.pose = bbox.center

            marker.scale = bbox.size

            # --------------------------------------------------
            # Green GT box
            # --------------------------------------------------

            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.5

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

        # ======================================================
        # PUBLISH
        # ======================================================

        self.gt_pub.publish(
            detections_msg
        )

        self.gt_vis_pub.publish(
            marker_array
        )

    # ==========================================================
    # CREATE POINTCLOUD2
    # ==========================================================

    def create_lidar_cloud_msg(
        self,
        points,
        header
    ):

        """
        Create a PointCloud2 containing:

            x
            y
            z
            intensity
            timestamp

        points shape:

            (N, 5)

        where:

            points[:, 0] = x
            points[:, 1] = y
            points[:, 2] = z
            points[:, 3] = intensity
            points[:, 4] = timestamp
        """

        # ======================================================
        # POINT FIELDS
        # ======================================================

        fields = [

            PointField(
                name="x",
                offset=0,
                datatype=PointField.FLOAT32,
                count=1
            ),

            PointField(
                name="y",
                offset=4,
                datatype=PointField.FLOAT32,
                count=1
            ),

            PointField(
                name="z",
                offset=8,
                datatype=PointField.FLOAT32,
                count=1
            ),

            PointField(
                name="intensity",
                offset=12,
                datatype=PointField.FLOAT32,
                count=1
            ),

            PointField(
                name="timestamp",
                offset=16,
                datatype=PointField.FLOAT32,
                count=1
            ),
        ]

        # ======================================================
        # CREATE MESSAGE
        # ======================================================

        cloud_msg = point_cloud2.create_cloud(
            header,
            fields,
            points
        )

        return cloud_msg

    # ==========================================================
    # TIMER CALLBACK
    # ==========================================================

    def timer_callback(self):

        # ======================================================
        # SCENE LOOP
        # ======================================================

        if self.sample_token == "":

            self.get_logger().info(
                "Restarting scene"
            )

            self.sample_token = (
                self.first_sample_token
            )

            return

        # ======================================================
        # CURRENT SAMPLE
        # ======================================================

        sample = self.nusc.get(
            "sample",
            self.sample_token
        )

        now = self.get_clock().now().to_msg()

        # ======================================================
        # IMAGE
        # ======================================================

        cam_data = self.nusc.get(
            "sample_data",
            sample["data"][self.cam]
        )

        img_path = os.path.join(
            self.nusc_root,
            cam_data["filename"]
        )

        img = cv2.imread(
            img_path
        )

        if img is None:

            self.get_logger().error(
                f"Could not read image: {img_path}"
            )

        else:

            img_msg = (
                self.bridge.cv2_to_imgmsg(
                    img,
                    encoding="bgr8"
                )
            )

            img_msg.header.stamp = now

            img_msg.header.frame_id = (
                "camera_front"
            )

            self.image_pub.publish(
                img_msg
            )

        # ======================================================
        # LIDAR
        # ======================================================
        #
        # IMPORTANT:
        #
        # The old implementation did:
        #
        #     LidarPointCloud.from_file()
        #
        # and then:
        #
        #     pc.points[:3, :]
        #
        # which published only:
        #
        #     x,y,z
        #
        # We now use:
        #
        #     from_file_multisweep()
        #
        # to construct the same type of representation expected
        # by the NuScenes OpenPCDet configuration:
        #
        #     x,y,z,intensity,timestamp
        #
        # using 10 sweeps.
        #
        # ======================================================

        try:

            pc, times = (
                LidarPointCloud.from_file_multisweep(
                    self.nusc,
                    sample,
                    self.lidar,
                    self.lidar,
                    nsweeps=self.num_lidar_sweeps
                )
            )

        except Exception as e:

            self.get_logger().error(
                "Failed to load LiDAR sweeps: "
                f"{e}"
            )

            return

        # ======================================================
        # CHECK POINT DATA
        # ======================================================

        # pc.points:
        #
        #     shape = (4, N)
        #
        #     [x]
        #     [y]
        #     [z]
        #     [intensity]
        #
        # times:
        #
        #     shape = (1, N)
        #
        #     time lag for every point
        #

        if pc.points.shape[1] == 0:

            self.get_logger().warn(
                "No LiDAR points found"
            )

            return

        # ======================================================
        # COMBINE FEATURES
        # ======================================================

        points = np.vstack(
            [
                pc.points,
                times
            ]
        ).T.astype(
            np.float32
        )

        # ======================================================
        # REMOVE INVALID VALUES
        # ======================================================

        finite_mask = np.all(
            np.isfinite(points),
            axis=1
        )

        points = points[
            finite_mask
        ]

        # ======================================================
        # LOG
        # ======================================================

        self.get_logger().info(
            "LiDAR:"
            f" sweeps={self.num_lidar_sweeps}"
            f" points={points.shape[0]}"
            f" features={points.shape[1]}"
        )

        # ======================================================
        # LIDAR HEADER
        # ======================================================

        lidar_header = (
            self.bridge.cv2_to_imgmsg(
                np.zeros(
                    (1, 1),
                    dtype=np.uint8
                ),
                encoding="mono8"
            ).header
        )

        # We don't actually need to construct an image here.
        # Use the current ROS timestamp directly.
        lidar_header.stamp = now

        lidar_header.frame_id = (
            "lidar_top"
        )

        # ======================================================
        # CREATE POINTCLOUD2
        # ======================================================

        cloud_msg = (
            self.create_lidar_cloud_msg(
                points,
                lidar_header
            )
        )

        # ======================================================
        # PUBLISH
        # ======================================================

        self.lidar_pub.publish(
            cloud_msg
        )

        # ======================================================
        # TF
        # ======================================================

        self.publish_tf(
            sample,
            now
        )

        # ======================================================
        # GROUND TRUTH
        # ======================================================

        self.publish_gt_boxes(
            sample,
            now
        )

        # ======================================================
        # NEXT SAMPLE
        # ======================================================

        self.sample_token = (
            sample["next"]
        )


# ==============================================================
# MAIN
# ==============================================================

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

        node.destroy_node()

        rclpy.shutdown()


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":

    main()