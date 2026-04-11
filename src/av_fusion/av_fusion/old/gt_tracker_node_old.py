import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from vision_msgs.msg import Detection3DArray, Detection3D
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np


class GTTracker(Node):

    def __init__(self):
        super().__init__("gt_tracker")

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        # ===== SUBSCRIBE (GREEN GT BOXES) =====
        self.sub = self.create_subscription(
            Detection3DArray,
            "/detections/gt_boxes",
            self.callback,
            qos
        )

        # ===== PUBLISH TRACKED BOXES =====
        self.track_pub = self.create_publisher(
            Detection3DArray,
            "/detections/tracked_boxes",
            qos
        )

        # ===== PUBLISH RED MARKERS =====
        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/gt/track_markers",
            qos
        )

        self.get_logger().info("🚀 GT Tracker Started")

    # ---------------------------------------------------------
    def callback(self, msg):

        # ---------- publish tracked detections ----------
        tracked_msg = Detection3DArray()
        tracked_msg.header = msg.header

        marker_array = MarkerArray()

        for i, det in enumerate(msg.detections):

            # ===== copy detection =====
            new_det = Detection3D()
            new_det.header = det.header
            new_det.bbox = det.bbox

            tracked_msg.detections.append(new_det)

            # ===== RED MARKER =====
            m = Marker()
            m.header = msg.header
            m.ns = "tracked_boxes"
            m.id = i

            m.type = Marker.CUBE
            m.action = Marker.ADD

            m.pose = det.bbox.center
            m.scale = det.bbox.size

            # RED color
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 0.6

            marker_array.markers.append(m)

        # publish both
        self.track_pub.publish(tracked_msg)
        self.marker_pub.publish(marker_array)


# ---------------------------------------------------------
def main():
    rclpy.init()
    node = GTTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
