import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

import math


class GTTracker(Node):

    def __init__(self):
        super().__init__("gt_tracker")

        # ---------------- QoS MATCHING PLAYER ----------------
        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        # Subscribe to GT detection markers (green boxes)
        self.subscription = self.create_subscription(
            MarkerArray,
            "/gt/detection_markers",
            self.detection_callback,
            qos
        )

        # Publish tracked markers (red boxes)
        self.track_marker_pub = self.create_publisher(
            MarkerArray,
            "/gt/track_markers",
            qos
        )

        self.get_logger().info("🚀 GT Tracker Started")

    # --------------------------------------------------------
    def detection_callback(self, msg: MarkerArray):

        track_array = MarkerArray()

        for marker in msg.markers:

            track_marker = Marker()

            track_marker.header = marker.header
            track_marker.ns = "gt_tracks"
            track_marker.id = marker.id
            track_marker.type = Marker.CUBE
            track_marker.action = Marker.ADD

            # Copy pose
            track_marker.pose = marker.pose

            # Copy size
            track_marker.scale = marker.scale

            # RED color for tracks
            track_marker.color = ColorRGBA()
            track_marker.color.r = 1.0
            track_marker.color.g = 0.0
            track_marker.color.b = 0.0
            track_marker.color.a = 0.6

            track_array.markers.append(track_marker)

        self.track_marker_pub.publish(track_array)


def main():
    rclpy.init()
    node = GTTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
