import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from visualization_msgs.msg import MarkerArray


class LidarDetectionVisualizer(Node):

    def __init__(self):
        super().__init__("lidar_detection_visualizer")

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        self.sub = self.create_subscription(
            MarkerArray,
            "/lidar/detection_markers",
            self.callback,
            qos
        )

        self.get_logger().info("Visualizer started (QoS FIXED)")

    def callback(self, msg):
        # Just forward to RViz
        self.get_logger().debug("Received markers")


def main():
    rclpy.init()
    node = LidarDetectionVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()