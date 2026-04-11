import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

class LidarDetectionVisualizer(Node):

    def __init__(self):
        super().__init__("lidar_detection_visualizer")

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        self.sub = self.create_subscription(
            Detection3DArray,
            "/detections/tracked_boxes",
            self.callback,
            qos
        )

        self.pub = self.create_publisher(
            MarkerArray,
            "/lidar/detection_markers",
            qos
        )

        self.get_logger().info("🚀 Lidar Detection Visualizer started")

    # ---------------------------------------------------------
    def callback(self, msg: Detection3DArray):

        marker_array = MarkerArray()

        for i, det in enumerate(msg.detections):

            marker = Marker()
            marker.header = msg.header
            marker.ns = "lidar_predictions"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # ---------------- POSITION ----------------
            marker.pose.position.x = det.bbox.center.position.x
            marker.pose.position.y = det.bbox.center.position.y
            marker.pose.position.z = det.bbox.center.position.z

            # ---------------- ORIENTATION ----------------
            marker.pose.orientation = det.bbox.center.orientation

            # ---------------- SIZE ----------------
            marker.scale.x = det.bbox.size.x
            marker.scale.y = det.bbox.size.y
            marker.scale.z = det.bbox.size.z

            # ---------------- COLOR (RED) ----------------
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.6

            # Auto-clear after 0.2 sec
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 200_000_000

            marker_array.markers.append(marker)

        self.pub.publish(marker_array)


def main():
    rclpy.init()
    node = LidarDetectionVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
