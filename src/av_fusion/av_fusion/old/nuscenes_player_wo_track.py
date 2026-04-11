import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import tf2_ros

import cv2
import numpy as np
import os

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from pyquaternion import Quaternion
from sensor_msgs_py import point_cloud2


class NuScenesAVPlayer(Node):

    def __init__(self):
        super().__init__("nuscenes_av_player")

        # ---------------- CONFIG ----------------
        self.nusc_root = "/home/adarsh/av_perception/data/nuscenes"
        self.version = "v1.0-mini"
        self.cam = "CAM_FRONT"
        self.lidar = "LIDAR_TOP"

        # ---------------- QoS ----------------
        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        self.bridge = CvBridge()

        self.image_pub = self.create_publisher(
            Image, "/camera/front/image", qos
        )

        self.lidar_pub = self.create_publisher(
            PointCloud2, "/lidar/points", qos
        )

        self.marker_pub = self.create_publisher(
            MarkerArray, "/detections/gt_boxes", qos
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

        # ---- Camera TF
        cam_cs = self.nusc.get(
            "calibrated_sensor",
            cam_data["calibrated_sensor_token"]
        )

        t2 = TransformStamped()
        t2.header.stamp = timestamp
        t2.header.frame_id = "base_link"
        t2.child_frame_id = "camera_front"

        t2.transform.translation.x = cam_cs["translation"][0]
        t2.transform.translation.y = cam_cs["translation"][1]
        t2.transform.translation.z = cam_cs["translation"][2]

        q = cam_cs["rotation"]
        t2.transform.rotation.x = q[1]
        t2.transform.rotation.y = q[2]
        t2.transform.rotation.z = q[3]
        t2.transform.rotation.w = q[0]

        self.tf_broadcaster.sendTransform(t2)

        # ---- LiDAR TF
        lidar_data = self.nusc.get("sample_data", sample["data"][self.lidar])
        lidar_cs = self.nusc.get(
            "calibrated_sensor",
            lidar_data["calibrated_sensor_token"]
        )

        t3 = TransformStamped()
        t3.header.stamp = timestamp
        t3.header.frame_id = "base_link"
        t3.child_frame_id = "lidar_top"

        t3.transform.translation.x = lidar_cs["translation"][0]
        t3.transform.translation.y = lidar_cs["translation"][1]
        t3.transform.translation.z = lidar_cs["translation"][2]

        q = lidar_cs["rotation"]
        t3.transform.rotation.x = q[1]
        t3.transform.rotation.y = q[2]
        t3.transform.rotation.z = q[3]
        t3.transform.rotation.w = q[0]

        self.tf_broadcaster.sendTransform(t3)

    # ---------------------------------------------------
    def publish_gt_boxes(self, sample, timestamp):

        marker_array = MarkerArray()

        for i, ann_token in enumerate(sample["anns"]):
            ann = self.nusc.get("sample_annotation", ann_token)

            box = Box(
                ann["translation"],
                ann["size"],
                Quaternion(ann["rotation"])   # FIXED
            )

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = timestamp
            marker.ns = "gt_boxes"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = box.center[0]
            marker.pose.position.y = box.center[1]
            marker.pose.position.z = box.center[2]

            marker.pose.orientation.x = box.orientation.x
            marker.pose.orientation.y = box.orientation.y
            marker.pose.orientation.z = box.orientation.z
            marker.pose.orientation.w = box.orientation.w

            marker.scale.x = box.wlh[1]
            marker.scale.y = box.wlh[0]
            marker.scale.z = box.wlh[2]

            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.5

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    # ---------------------------------------------------
    def timer_callback(self):

        if self.sample_token == "":
            self.get_logger().info("🔁 Restarting scene")
            self.sample_token = self.first_sample_token
            return

        sample = self.nusc.get("sample", self.sample_token)
        now = self.get_clock().now().to_msg()

        # ---------------- IMAGE ----------------
        cam_data = self.nusc.get("sample_data", sample["data"][self.cam])
        img_path = os.path.join(self.nusc_root, cam_data["filename"])
        img = cv2.imread(img_path)

        img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        img_msg.header.stamp = now
        img_msg.header.frame_id = "camera_front"
        self.image_pub.publish(img_msg)

        # ---------------- LIDAR ----------------
        lidar_data = self.nusc.get("sample_data", sample["data"][self.lidar])
        pc = LidarPointCloud.from_file(
            os.path.join(self.nusc_root, lidar_data["filename"])
        )

        points = pc.points[:3, :].T.astype(np.float32)

        header = img_msg.header
        header.frame_id = "lidar_top"

        cloud_msg = point_cloud2.create_cloud_xyz32(header, points)
        self.lidar_pub.publish(cloud_msg)

        # ---------------- TF ----------------
        self.publish_tf(sample, now)

        # ---------------- GT BOXES ----------------
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
