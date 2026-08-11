import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs_py import point_cloud2
import numpy as np
import torch
from mmdet3d.apis import init_model, inference_detector

CONFIG = "configs/pointpillars_config.py"
CHECKPOINT = "checkpoints/pointpillars.pth"


class PointPillarsDetector(Node):

    def __init__(self):
        super().__init__("pointpillars_detector")

        self.subscription = self.create_subscription(
            PointCloud2,
            "/lidar/points",
            self.lidar_callback,
            10
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/detections/markers",
            10
        )

        # Dynamic device selection for CPU-only systems
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = init_model(CONFIG, CHECKPOINT, device=device)

        self.get_logger().info(f"PointPillars Detector Started on device: {device}")

    def lidar_callback(self, msg):
        points = []

        # Read intensity along with x, y, z for point pillars
        for p in point_cloud2.read_points(
                msg,
                field_names=("x", "y", "z", "intensity"),
                skip_nans=True):
            points.append([p[0], p[1], p[2], p[3]])

        if len(points) == 0:
            return

        points = np.array(points)

        result, _ = inference_detector(self.model, points)

        detections = Detection3DArray()
        detections.header = msg.header
        marker_array = MarkerArray()

        boxes = result.pred_instances_3d.bboxes_3d

        for idx, box in enumerate(boxes):
            det = Detection3D()
            bbox = BoundingBox3D()

            center = box.center.numpy()
            dims = box.dims.numpy()

            bbox.center.position.x = float(center[0])
            bbox.center.position.y = float(center[1])
            bbox.center.position.z = float(center[2])

            bbox.size.x = float(dims[0])
            bbox.size.y = float(dims[1])
            bbox.size.z = float(dims[2])

            det.bbox = bbox
            detections.detections.append(det)

            # RViz Marker
            marker = Marker()
            marker.header = msg.header
            marker.ns = "pointpillars_detections"
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = float(center[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(dims[0])
            marker.scale.y = float(dims[1])
            marker.scale.z = float(dims[2])
            marker.color.r = 0.0
            marker.color.g = 0.8
            marker.color.b = 1.0
            marker.color.a = 0.5
            marker.lifetime.sec = 1

            marker_array.markers.append(marker)

        self.publisher.publish(detections)
        self.marker_pub.publish(marker_array)


def main():
    rclpy.init()
    node = PointPillarsDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()