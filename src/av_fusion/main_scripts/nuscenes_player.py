import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D

from cv_bridge import CvBridge
import tf2_ros

import cv2
import numpy as np
import os

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from sensor_msgs_py import point_cloud2


class NuScenesAVPlayer(Node):

    def __init__(self):
        super().__init__("nuscenes_av_player")

        # ---------------- CONFIG ----------------
        self.nusc_root = "/home/adarsh/av_perception/data/nuscenes"
        self.version = "v1.0-mini"
        self.cam = "CAM_FRONT"
        self.lidar = "LIDAR_TOP"

        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        self.bridge = CvBridge()

        # ---------------- Publishers ----------------
        self.image_pub = self.create_publisher(
            Image, "/camera/front/image", qos
        )

        self.lidar_pub = self.create_publisher(
            PointCloud2, "/lidar/points", qos
        )

        # Structured GT (for pipeline)
        self.gt_pub = self.create_publisher(
            Detection3DArray,
            "/detections/gt_boxes",
            qos
        )

        # Visualization GT (for RViz)
        self.gt_vis_pub = self.create_publisher(
            MarkerArray,
            "/gt/visualization_markers",
            qos
        )

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # ---------------- NUSCENES ----------------
        self.nusc = NuScenes(
            version=self.version,
            dataroot=self.nusc_root,
            verbose=False
        )

        self.scene = self.nusc.scene[0]
        self.first_sample_token = self.scene["first_sample_token"]
        self.sample_token = self.first_sample_token

        self.timer = self.create_timer(0.5, self.timer_callback)

        self.get_logger().info("🚀 NuScenes AV Player Started")

    # ---------------------------------------------------
    def publish_tf(self, sample, timestamp):

        cam_data = self.nusc.get("sample_data", sample["data"][self.cam])
        ego_pose = self.nusc.get("ego_pose", cam_data["ego_pose_token"])

        # map → base_link
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"

        t.transform.translation.x = ego_pose["translation"][0]
        t.transform.translation.y = ego_pose["translation"][1]
        t.transform.translation.z = ego_pose["translation"][2]

        q = ego_pose["rotation"]
        t.transform.rotation.x = q[1]
        t.transform.rotation.y = q[2]
        t.transform.rotation.z = q[3]
        t.transform.rotation.w = q[0]

        self.tf_broadcaster.sendTransform(t)

        # base_link → lidar_top
        lidar_data = self.nusc.get("sample_data", sample["data"][self.lidar])
        lidar_cs = self.nusc.get(
            "calibrated_sensor",
            lidar_data["calibrated_sensor_token"]
        )

        t2 = TransformStamped()
        t2.header.stamp = timestamp
        t2.header.frame_id = "base_link"
        t2.child_frame_id = "lidar_top"

        t2.transform.translation.x = lidar_cs["translation"][0]
        t2.transform.translation.y = lidar_cs["translation"][1]
        t2.transform.translation.z = lidar_cs["translation"][2]

        q = lidar_cs["rotation"]
        t2.transform.rotation.x = q[1]
        t2.transform.rotation.y = q[2]
        t2.transform.rotation.z = q[3]
        t2.transform.rotation.w = q[0]

        self.tf_broadcaster.sendTransform(t2)

    # ---------------------------------------------------
    def publish_gt_boxes(self, sample, timestamp):

        detections_msg = Detection3DArray()
        detections_msg.header.frame_id = "map"
        detections_msg.header.stamp = timestamp

        marker_array = MarkerArray()

        for i, ann_token in enumerate(sample["anns"]):
            ann = self.nusc.get("sample_annotation", ann_token)

            detection = Detection3D()
            detection.header = detections_msg.header

            bbox = BoundingBox3D()

            # Position
            bbox.center.position.x = float(ann["translation"][0])
            bbox.center.position.y = float(ann["translation"][1])
            bbox.center.position.z = float(ann["translation"][2])

            # Orientation
            q = ann["rotation"]
            bbox.center.orientation.x = float(q[1])
            bbox.center.orientation.y = float(q[2])
            bbox.center.orientation.z = float(q[3])
            bbox.center.orientation.w = float(q[0])

            # Size (nuScenes wlh)
            bbox.size.x = float(ann["size"][1])  # length
            bbox.size.y = float(ann["size"][0])  # width
            bbox.size.z = float(ann["size"][2])  # height

            detection.bbox = bbox
            detections_msg.detections.append(detection)

            # -------- GREEN VISUALIZATION --------
            m = Marker()
            m.header = detections_msg.header
            m.ns = "gt_boxes"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose = bbox.center
            m.scale = bbox.size

            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 0.5

            marker_array.markers.append(m)

        self.gt_pub.publish(detections_msg)
        self.gt_vis_pub.publish(marker_array)

    # ---------------------------------------------------
    def timer_callback(self):

        if self.sample_token == "":
            self.get_logger().info("🔁 Restarting scene")
            self.sample_token = self.first_sample_token
            return

        sample = self.nusc.get("sample", self.sample_token)
        now = self.get_clock().now().to_msg()

        # -------- IMAGE --------
        cam_data = self.nusc.get("sample_data", sample["data"][self.cam])
        img_path = os.path.join(self.nusc_root, cam_data["filename"])
        img = cv2.imread(img_path)

        img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        img_msg.header.stamp = now
        img_msg.header.frame_id = "camera_front"
        self.image_pub.publish(img_msg)

        # -------- LIDAR --------
        lidar_data = self.nusc.get("sample_data", sample["data"][self.lidar])
        pc = LidarPointCloud.from_file(
            os.path.join(self.nusc_root, lidar_data["filename"])
        )

        points = pc.points[:3, :].T.astype(np.float32)

        cloud_header = img_msg.header
        cloud_header.frame_id = "lidar_top"

        cloud_msg = point_cloud2.create_cloud_xyz32(
            cloud_header, points
        )

        self.lidar_pub.publish(cloud_msg)

        # -------- TF --------
        self.publish_tf(sample, now)

        # -------- GT --------
        self.publish_gt_boxes(sample, now)

        self.sample_token = sample["next"]


def main():
    rclpy.init()
    node = NuScenesAVPlayer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()