
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
from std_msgs.msg import String
import math


class DecisionNode(Node):

    def __init__(self):
        super().__init__("decision_node")

        self.sub = self.create_subscription(
            Detection3DArray,
            "/tracked_objects",
            self.callback,
            10
        )

        self.pub = self.create_publisher(
            String,
            "/adas/decision",
            10
        )

        self.get_logger().info("🚀 ADAS Decision Node Started")

    def callback(self, msg):
        decision = "SAFE"

        for det in msg.detections:
            x = det.bbox.center.position.x
            y = det.bbox.center.position.y

            # Extract velocity from results array
            if len(det.results) > 0:
                vx = det.results[0].pose.pose.position.x
                vy = det.results[0].pose.pose.position.y
            else:
                vx = 0.0
                vy = 0.0

            dist = math.sqrt(x**2 + y**2)

            # Radial relative velocity along line connecting ego (0,0) to object (x,y)
            # rel_vel is negative if the object is approaching ego
            rel_vel = (x*vx + y*vy) / (dist + 1e-6)

            if rel_vel < 0:
                ttc = dist / (-rel_vel + 1e-6)
            else:
                ttc = float("inf")

            # 🔥 ADAS LOGIC
            if ttc < 2.0:
                decision = "BRAKE"
                break
            elif ttc < 4.0:
                decision = "SLOW"

        msg_out = String()
        msg_out.data = decision

        self.pub.publish(msg_out)
        self.get_logger().info(f"🚗 Decision: {decision}")


def main():
    rclpy.init()
    node = DecisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()


