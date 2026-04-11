
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray


class CameraDetector(Node):

    def __init__(self):
        super().__init__("camera_detector")

        self.pub = self.create_publisher(
            Detection2DArray,
            "/detections/boxes_2d",
            10
        )

        self.get_logger().info("🚀 Camera Detector Placeholder Started")

    def timer_callback(self):
        msg = Detection2DArray()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = CameraDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

