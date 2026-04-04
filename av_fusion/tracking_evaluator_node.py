import rclpy
from rclpy.node import Node

from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import MarkerArray

import numpy as np


class TrackingEvaluator(Node):

    def __init__(self):
        super().__init__("tracking_evaluator")

        self.gt_sub = self.create_subscription(
            Detection3DArray,
            "/detections/boxes_3d",
            self.gt_callback,
            10
        )

        self.trk_sub = self.create_subscription(
            MarkerArray,
            "/lidar/detection_markers",
            self.trk_callback,
            10
        )

        self.gt_data = None
        self.trk_data = None

        self.dist_thresh = 2.0

        self.frame_count = 0
        self.total_matches = 0
        self.total_gt = 0
        self.total_pred = 0
        self.total_error = 0.0

        self.get_logger().info("📊 Tracking Evaluator Started")

    # ---------------- GT ----------------
    def gt_callback(self, msg):
        self.gt_data = [
            [det.bbox.center.position.x, det.bbox.center.position.y]
            for det in msg.detections
        ]

    # ---------------- TRACKS ----------------
    def trk_callback(self, msg):
        self.trk_data = [
            [m.pose.position.x, m.pose.position.y]
            for m in msg.markers
            if m.ns == "tracked"
        ]

        self.evaluate()

    # ---------------- EVALUATION ----------------
    def evaluate(self):

        if self.gt_data is None or self.trk_data is None:
            return

        gt = np.array(self.gt_data)
        trk = np.array(self.trk_data)

        if len(gt) == 0 or len(trk) == 0:
            return

        self.frame_count += 1

        matches = 0
        errors = []

        used_trk = set()

        for g in gt:

            best_dist = float("inf")
            best_j = -1

            for j, t in enumerate(trk):

                if j in used_trk:
                    continue

                dist = np.linalg.norm(g - t)

                if dist < best_dist:
                    best_dist = dist
                    best_j = j

            if best_dist < self.dist_thresh:
                matches += 1
                used_trk.add(best_j)
                errors.append(best_dist)

        precision = matches / len(trk)
        recall = matches / len(gt)
        avg_error = np.mean(errors) if len(errors) > 0 else 0.0

        self.total_matches += matches
        self.total_gt += len(gt)
        self.total_pred += len(trk)
        self.total_error += sum(errors)

        # Print every 10 frames
        if self.frame_count % 10 == 0:

            overall_precision = self.total_matches / max(self.total_pred, 1)
            overall_recall = self.total_matches / max(self.total_gt, 1)
            overall_error = self.total_error / max(self.total_matches, 1)

            self.get_logger().info(
                f"\n📊 Frame {self.frame_count}\n"
                f"Precision: {precision:.2f} | Recall: {recall:.2f} | Error: {avg_error:.2f}\n"
                f"Overall → Precision: {overall_precision:.2f}, Recall: {overall_recall:.2f}, Error: {overall_error:.2f}\n"
            )


def main():
    rclpy.init()
    node = TrackingEvaluator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()