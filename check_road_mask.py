import rclpy
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image


class RoadMaskChecker(Node):

    def __init__(self):
        super().__init__("road_mask_checker")

        self.sub = self.create_subscription(
            Image,
            "/camera/segmentation/road_mask",
            self.callback,
            10
        )

        self.done = False

    def callback(self, msg):

        data = np.frombuffer(
            bytes(msg.data),
            dtype=np.uint8
        )

        mask = data.reshape(
            msg.height,
            msg.width
        )

        nonzero = np.count_nonzero(mask)

        total = mask.size

        percentage = (
            100.0 * nonzero / total
        )

        unique, counts = np.unique(
            mask,
            return_counts=True
        )

        self.get_logger().info(
            f"Mask shape: {mask.shape}"
        )

        self.get_logger().info(
            f"Non-zero pixels: "
            f"{nonzero}/{total} "
            f"({percentage:.2f}%)"
        )

        self.get_logger().info(
            f"Min: {mask.min()} | "
            f"Max: {mask.max()}"
        )

        self.get_logger().info(
            f"Unique values: "
            f"{dict(zip(unique.tolist(), counts.tolist()))}"
        )

        self.done = True


def main():

    rclpy.init()

    node = RoadMaskChecker()

    while rclpy.ok() and not node.done:
        rclpy.spin_once(
            node,
            timeout_sec=1.0
        )

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
