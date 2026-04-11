import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3D, Detection3DArray, BoundingBox3D

import numpy as np
from sklearn.cluster import DBSCAN
from sensor_msgs_py import point_cloud2

import tf2_ros
from geometry_msgs.msg import PointStamped
from rclpy.time import Time
from rclpy.duration import Duration


class LiDARClusterDetector(Node):

    def __init__(self):
        super().__init__("lidar_cluster_detector")

        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        self.subscription = self.create_subscription(
            PointCloud2,
            "/lidar/points",
            self.lidar_callback,
            qos
        )

        self.publisher = self.create_publisher(
            Detection3DArray,
            "/lidar/detections",
            10
        )

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info("🚀 LiDAR Cluster Detector (Map Frame) Started")

    # -------------------------------------------------------
    def lidar_callback(self, msg):

        # Check TF availability first
        if not self.tf_buffer.can_transform(
            "map",
            msg.header.frame_id,
            Time(),
            timeout=Duration(seconds=0.1)
        ):
            self.get_logger().warn("TF not ready yet")
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                "map",
                msg.header.frame_id,
                Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        # -------------------------------
        # Convert point cloud to numpy
        # -------------------------------
        points = []

        for p in point_cloud2.read_points(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=True):
            points.append([p[0], p[1], p[2]])

        if len(points) == 0:
            return

        points = np.array(points)

        # Ground removal
        points = points[points[:, 2] > -1.5]

        # Clustering
        clustering = DBSCAN(eps=0.8, min_samples=15).fit(points)
        labels = clustering.labels_

        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)

        detections_msg = Detection3DArray()
        detections_msg.header.frame_id = "map"
        detections_msg.header.stamp = msg.header.stamp

        for label in unique_labels:

            cluster_points = points[labels == label]
            if cluster_points.shape[0] < 20:
                continue

            min_pt = np.min(cluster_points, axis=0)
            max_pt = np.max(cluster_points, axis=0)

            center = (min_pt + max_pt) / 2.0
            size = max_pt - min_pt

            # Transform center to map frame
            pt = PointStamped()
            pt.header = msg.header
            pt.point.x = float(center[0])
            pt.point.y = float(center[1])
            pt.point.z = float(center[2])

            try:
                transformed = tf2_ros.do_transform_point(pt, transform)
            except Exception:
                continue

            detection = Detection3D()
            detection.header = detections_msg.header

            bbox = BoundingBox3D()
            bbox.center.position.x = transformed.point.x
            bbox.center.position.y = transformed.point.y
            bbox.center.position.z = transformed.point.z

            bbox.size.x = float(size[0])
            bbox.size.y = float(size[1])
            bbox.size.z = float(size[2])

            detection.bbox = bbox
            detections_msg.detections.append(detection)

        self.publisher.publish(detections_msg)


def main():
    rclpy.init()
    node = LiDARClusterDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
