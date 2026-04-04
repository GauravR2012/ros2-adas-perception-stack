import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D

from sensor_msgs_py import point_cloud2
import numpy as np

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

        self.publisher = self.create_publisher(
            Detection3DArray,
            "/detections/3d_boxes",
            10
        )

        self.model = init_model(CONFIG, CHECKPOINT, device="cuda")

        self.get_logger().info("PointPillars Detector Started")

    def lidar_callback(self, msg):

        points = []

        for p in point_cloud2.read_points(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=True):

            points.append([p[0], p[1], p[2]])

        points = np.array(points)

        result, _ = inference_detector(self.model, points)

        detections = Detection3DArray()
        detections.header = msg.header

        boxes = result.pred_instances_3d.bboxes_3d

        for box in boxes:

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

        self.publisher.publish(detections)


def main():
    rclpy.init()
    node = PointPillarsDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()