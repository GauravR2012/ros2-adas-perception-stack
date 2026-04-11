import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
from scipy.optimize import linear_sum_assignment


# ==============================
# Kalman Tracker
# ==============================
class KalmanTracker:

    count = 0

    def __init__(self, x, y):
        # State: [x, y, vx, vy]
        self.x = np.array([[x], [y], [0.0], [0.0]])
        self.P = np.eye(4) * 10.0

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        self.R = np.eye(2) * 1.0
        self.Q = np.eye(4) * 0.05

        self.id = KalmanTracker.count
        KalmanTracker.count += 1

        self.missed = 0

    def predict(self, dt):

        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        z = np.array([[z[0]], [z[1]]])

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.missed = 0


# ==============================
# Tracker Node
# ==============================
class GTTracker(Node):

    def __init__(self):
        super().__init__("gt_tracker")

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        self.sub = self.create_subscription(
            Detection3DArray,
            "/detections/gt_boxes",
            self.callback,
            qos
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/lidar/detection_markers",
            qos
        )

        self.trackers = []
        self.max_missed = 5
        self.dist_threshold = 3.0

        self.last_time = None

        self.get_logger().info("Kalman Tracker with Velocity Started")

    # ----------------------------------------
    def callback(self, msg):

        current_time = (
            msg.header.stamp.sec +
            msg.header.stamp.nanosec * 1e-9
        )

        if self.last_time is None:
            self.last_time = current_time
            return

        dt = current_time - self.last_time
        self.last_time = current_time

        # Extract detections
        detections = []
        for det in msg.detections:
            x = det.bbox.center.position.x
            y = det.bbox.center.position.y
            detections.append([x, y])

        detections = np.array(detections)

        # Predict
        predictions = []
        for trk in self.trackers:
            trk.predict(dt)
            predictions.append([trk.x[0, 0], trk.x[1, 0]])

        if len(predictions) == 0:
            predictions = np.empty((0, 2))
        else:
            predictions = np.array(predictions)

        # Association
        if len(detections) > 0 and len(predictions) > 0:

            cost_matrix = np.linalg.norm(
                detections[:, None, :] - predictions[None, :, :],
                axis=2
            )

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assigned_tracks = set()
            assigned_dets = set()

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < self.dist_threshold:
                    self.trackers[c].update(detections[r])
                    assigned_tracks.add(c)
                    assigned_dets.add(r)

            # Unmatched trackers
            for i, trk in enumerate(self.trackers):
                if i not in assigned_tracks:
                    trk.missed += 1

            # New trackers
            for i, det in enumerate(detections):
                if i not in assigned_dets:
                    self.trackers.append(
                        KalmanTracker(det[0], det[1])
                    )

        else:
            for det in detections:
                self.trackers.append(
                    KalmanTracker(det[0], det[1])
                )

        # Remove dead tracks
        self.trackers = [
            t for t in self.trackers if t.missed < self.max_missed
        ]

        # Publish markers
        marker_array = MarkerArray()

        for trk in self.trackers:

            x = trk.x[0, 0]
            y = trk.x[1, 0]
            vx = trk.x[2, 0]
            vy = trk.x[3, 0]
            speed = np.sqrt(vx**2 + vy**2)

            # -------- BOX --------
            m = Marker()
            m.header = msg.header
            m.ns = "tracked_boxes"
            m.id = trk.id
            m.type = Marker.CUBE
            m.action = Marker.ADD

            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.0

            m.scale.x = 2.0
            m.scale.y = 1.0
            m.scale.z = 1.5

            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 0.8

            marker_array.markers.append(m)

            # -------- VELOCITY TEXT --------
            text_marker = Marker()
            text_marker.header = msg.header
            text_marker.ns = "velocity_text"
            text_marker.id = 1000 + trk.id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = x
            text_marker.pose.position.y = y
            text_marker.pose.position.z = 3.0

            text_marker.scale.z = 0.8

            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 0.0
            text_marker.color.a = 1.0

            text_marker.text = f"{speed:.2f} m/s"

            marker_array.markers.append(text_marker)

        self.marker_pub.publish(marker_array)


def main():
    rclpy.init()
    node = GTTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()