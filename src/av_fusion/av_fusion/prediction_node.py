
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray


class PredictionNode(Node):

    def __init__(self):
        super().__init__("prediction_node")

        self.sub = self.create_subscription(
            Detection3DArray,
            "/tracked_objects",
            self.callback,
            10
        )

        self.pub = self.create_publisher(
            MarkerArray,
            "/predictions",
            10
        )

        self.get_logger().info("🚀 Prediction Node (REAL) Started")

    def callback(self, msg):
        marker_array = MarkerArray()

        for i, det in enumerate(msg.detections):
            x = det.bbox.center.position.x
            y = det.bbox.center.position.y

            # Extract velocity from results array
            if len(det.results) > 0:
                vx = det.results[0].pose.pose.position.x
                vy = det.results[0].pose.pose.position.y
                track_id = int(det.results[0].hypothesis.class_id)
            else:
                vx = 0.0
                vy = 0.0
                track_id = i

            for step in range(10):
                t = step * 0.5

                px = x + vx * t
                py = y + vy * t

                m = Marker()
                m.header = msg.header
                m.ns = "prediction"
                m.id = track_id * 100 + step
                m.type = Marker.SPHERE
                m.action = Marker.ADD

                m.pose.position.x = px
                m.pose.position.y = py
                m.pose.position.z = 0.5

                m.scale.x = 0.3
                m.scale.y = 0.3
                m.scale.z = 0.3

                m.color.r = 1.0
                m.color.g = 0.5  # Orange/yellow predictions to stand out
                m.color.b = 0.0
                m.color.a = 0.6

                m.lifetime.sec = 0
                m.lifetime.nanosec = 600000000

                marker_array.markers.append(m)

        self.pub.publish(marker_array)


def main():
    rclpy.init()
    node = PredictionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()



