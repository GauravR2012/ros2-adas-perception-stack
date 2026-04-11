import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from sensor_msgs_py import point_cloud2


class NuScenesPlayer(Node):

    def __init__(self):
        super().__init__("nuscenes_player")

        # ---------------- CONFIG ----------------
        self.nusc_root = "/home/adarsh/av_perception/data/nuscenes"
        self.version = "v1.0-mini"
        self.cam = "CAM_FRONT"
        self.lidar = "LIDAR_TOP"

        # ---------------- ROS QoS ----------------
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

        # ---------------- NUSCENES ----------------
        self.nusc = NuScenes(
            version=self.version,
            dataroot=self.nusc_root,
            verbose=False
        )

        self.scene = self.nusc.scene[0]
        self.first_sample_token = self.scene["first_sample_token"]
        self.sample_token = self.first_sample_token

        self.get_logger().info("NuScenes loaded")

        # ~2Hz publishing
        self.timer = self.create_timer(0.5, self.timer_callback)

        self.get_logger().info("✅ NuScenes Player Running (Loop Enabled)")

    # ---------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------
    def timer_callback(self):

        if self.sample_token == "":
            self.get_logger().info("🔁 End of scene — restarting")
            self.sample_token = self.first_sample_token
            return

        sample = self.nusc.get("sample", self.sample_token)

        # ================= IMAGE =================
        cam_data = self.nusc.get(
            "sample_data", sample["data"][self.cam]
        )

        img_path = os.path.join(
            self.nusc_root, cam_data["filename"]
        )

        img = cv2.imread(img_path)
        if img is None:
            self.get_logger().warn("Image not found")
            return

        img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = "camera_front"

        self.image_pub.publish(img_msg)

        # ================= LIDAR =================
        lidar_data = self.nusc.get(
            "sample_data", sample["data"][self.lidar]
        )

        pc = LidarPointCloud.from_file(
            os.path.join(self.nusc_root, lidar_data["filename"])
        )

        points = pc.points[:3, :].T.astype(np.float32)

        header = img_msg.header
        header.frame_id = "lidar_top"

        cloud_msg = point_cloud2.create_cloud_xyz32(
            header, points
        )

        self.lidar_pub.publish(cloud_msg)

        self.get_logger().info("📤 Frame published")

        # Move to next sample
        self.sample_token = sample["next"]


def main():
    rclpy.init()
    node = NuScenesPlayer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
