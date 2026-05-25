import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
from sensor_msgs_py import point_cloud2
from sklearn.cluster import DBSCAN


class LidarClusterDetector(Node):

    def __init__(self):

        super().__init__("lidar_cluster_detector")

        # ==========================================================
        # SUBSCRIBE
        # ==========================================================

        self.sub = self.create_subscription(
            PointCloud2,
            "/lidar/points",
            self.callback,
            10
        )

        # ==========================================================
        # PUBLISHERS
        # ==========================================================

        # Perception output
        self.detection_pub = self.create_publisher(
            Detection3DArray,
            "/detections/boxes_3d",
            10
        )

        # RViz visualization
        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/detections/markers",
            10
        )

        self.get_logger().info(
            "🚀 LiDAR Clustering Detector (RViz + Detection3D)"
        )

    # ==============================================================
    # CALLBACK
    # ==============================================================

    def callback(self, msg):

        self.get_logger().info("📡 Callback triggered")

        # ==========================================================
        # PointCloud2 → numpy
        # ==========================================================

        points = []

        for p in point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True
        ):
            points.append([p[0], p[1], p[2]])

        if len(points) == 0:
            self.get_logger().warn("⚠️ No points received")
            return

        points = np.array(points)

        self.get_logger().info(f"Raw points: {len(points)}")

        # ==========================================================
        # ROI FILTER
        # ==========================================================

        dist = np.linalg.norm(points[:, :2], axis=1)

        mask = (
            (dist < 45) &
            (points[:, 0] > -5) &
            (np.abs(points[:, 1]) < 20)
        )

        points = points[mask]

        self.get_logger().info(
            f"After range filter: {len(points)}"
        )

        # ==========================================================
        # GROUND REMOVAL (RANSAC)
        # ==========================================================

        if len(points) > 50:

            max_inliers = 0
            best_plane = None

            num_iterations = 40
            threshold = 0.25

            for _ in range(num_iterations):

                idx = np.random.choice(
                    len(points),
                    3,
                    replace=False
                )

                p1, p2, p3 = points[idx]

                normal = np.cross(p2 - p1, p3 - p1)

                norm = np.linalg.norm(normal)

                if norm < 1e-4:
                    continue

                normal = normal / norm

                d = -np.dot(normal, p1)

                distances = np.abs(
                    np.dot(points, normal) + d
                )

                inliers = np.where(
                    distances < threshold
                )[0]

                if len(inliers) > max_inliers:
                    max_inliers = len(inliers)
                    best_plane = (normal, d, inliers)

            if max_inliers > 0:

                normal, d, inliers = best_plane

                if abs(normal[2]) > 0.7:

                    mask_ground = np.zeros(
                        len(points),
                        dtype=bool
                    )

                    mask_ground[inliers] = True

                    mask_ground = (
                        mask_ground &
                        (points[:, 2] < -0.5)
                    )

                    points = points[~mask_ground]

        self.get_logger().info(
            f"After ground removal: {len(points)}"
        )

        # ==========================================================
        # DOWNSAMPLE
        # ==========================================================

        points = points[::2]

        # ==========================================================
        # DBSCAN CLUSTERING
        # ==========================================================

        clustering = DBSCAN(
            eps=1.2,
            min_samples=8
        ).fit(points)

        labels = clustering.labels_

        unique_labels = set(labels)

        # ==========================================================
        # OUTPUT MESSAGES
        # ==========================================================

        detections = Detection3DArray()
        detections.header = msg.header

        marker_array = MarkerArray()

        valid_clusters = 0

        # ==========================================================
        # PROCESS CLUSTERS
        # ==========================================================

        for lbl in unique_labels:

            if lbl == -1:
                continue

            cluster = points[labels == lbl]

            if len(cluster) < 10:
                continue

            min_bound = cluster.min(axis=0)
            max_bound = cluster.max(axis=0)

            center = (min_bound + max_bound) / 2
            size = max_bound - min_bound

            # ======================================================
            # SIZE FILTER
            # ======================================================

            if size[0] < 0.5 or size[1] < 0.5:
                continue

            if size[0] > 8 or size[1] > 8:
                continue

            # ======================================================
            # Detection3D
            # ======================================================

            det = Detection3D()

            det.header = msg.header

            bbox = BoundingBox3D()

            bbox.center.position.x = float(center[0])
            bbox.center.position.y = float(center[1])
            bbox.center.position.z = float(center[2])

            bbox.center.orientation.w = 1.0

            bbox.size.x = float(size[0])
            bbox.size.y = float(size[1])
            bbox.size.z = float(size[2])

            det.bbox = bbox

            detections.detections.append(det)

            # ======================================================
            # RViz Marker
            # ======================================================

            marker = Marker()

            marker.header = msg.header

            marker.ns = "lidar_clusters"

            marker.id = valid_clusters

            marker.type = Marker.CUBE

            marker.action = Marker.ADD

            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = float(center[2])

            marker.pose.orientation.w = 1.0

            marker.scale.x = float(size[0])
            marker.scale.y = float(size[1])
            marker.scale.z = float(size[2])

            # Green transparent boxes
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.4

            marker.lifetime.sec = 1

            marker_array.markers.append(marker)

            valid_clusters += 1

        # ==========================================================
        # PUBLISH
        # ==========================================================

        self.get_logger().info(
            f"✅ Publishing {valid_clusters} clusters"
        )

        self.detection_pub.publish(detections)

        self.marker_pub.publish(marker_array)


# ==============================================================
# MAIN
# ==============================================================

def main():

    rclpy.init()

    node = LidarClusterDetector()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()