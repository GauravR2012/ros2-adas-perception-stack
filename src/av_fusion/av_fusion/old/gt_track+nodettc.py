import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
from scipy.optimize import linear_sum_assignment

import tf2_ros
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point
from rclpy.time import Time
from rclpy.duration import Duration


# =========================================
# Kalman Tracker
# =========================================
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


# =========================================
# Tracker Node
# =========================================
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

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.trackers = []
        self.max_missed = 5
        self.dist_threshold = 3.0

        self.last_time = None

        self.get_logger().info("Kalman Tracker + Velocity + TTC Started")

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

        # ----------------------------
        # Extract detections
        # ----------------------------
        detections = []
        for det in msg.detections:
            x = det.bbox.center.position.x
            y = det.bbox.center.position.y
            detections.append([x, y])

        detections = np.array(detections)

        # ----------------------------
        # Predict
        # ----------------------------
        predictions = []
        for trk in self.trackers:
            trk.predict(dt)
            predictions.append([trk.x[0, 0], trk.x[1, 0]])

        if len(predictions) == 0:
            predictions = np.empty((0, 2))
        else:
            predictions = np.array(predictions)

        # ----------------------------
        # Association
        # ----------------------------
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

        # ----------------------------
        # Get TF map → base_link
        # ----------------------------
        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link",
                msg.header.frame_id,
                Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=0.1)
            )
        except:
            return

        # ----------------------------
        # Publish markers
        # ----------------------------
        marker_array = MarkerArray()

        for trk in self.trackers:

            x = trk.x[0, 0]
            y = trk.x[1, 0]
            vx = trk.x[2, 0]
            vy = trk.x[3, 0]
            speed = np.sqrt(vx**2 + vy**2)

            # Transform position to ego frame
            pt = PointStamped()
            pt.header = msg.header
            pt.point.x = x
            pt.point.y = y
            pt.point.z = 0.0

            ego_point = do_transform_point(pt, transform)

            long_dist = ego_point.point.x
            lat_dist = ego_point.point.y

            # Simplified relative velocity (ego assumed 0)
            v_rel = vx

            if long_dist > 0 and v_rel < 0:
                ttc = long_dist / (-v_rel)
            else:
                ttc = float("inf")

            # -------- BOX --------
            box = Marker()
            box.header = msg.header
            box.ns = "tracked_boxes"
            box.id = trk.id
            box.type = Marker.CUBE
            box.action = Marker.ADD

            box.pose.position.x = x
            box.pose.position.y = y
            box.pose.position.z = 0.0

            box.scale.x = 2.0
            box.scale.y = 1.0
            box.scale.z = 1.5

            box.color.r = 1.0
            box.color.g = 0.0
            box.color.b = 0.0
            box.color.a = 0.8

            marker_array.markers.append(box)

            # -------- VELOCITY TEXT --------
            vel_text = Marker()
            vel_text.header = msg.header
            vel_text.ns = "velocity_text"
            vel_text.id = 1000 + trk.id
            vel_text.type = Marker.TEXT_VIEW_FACING
            vel_text.action = Marker.ADD

            vel_text.pose.position.x = x
            vel_text.pose.position.y = y
            vel_text.pose.position.z = 3.0

            vel_text.scale.z = 0.8

            vel_text.color.r = 1.0
            vel_text.color.g = 1.0
            vel_text.color.b = 0.0
            vel_text.color.a = 1.0

            vel_text.text = f"{speed:.2f} m/s"

            marker_array.markers.append(vel_text)

            # -------- TTC TEXT --------
            ttc_text = Marker()
            ttc_text.header = msg.header
            ttc_text.ns = "ttc_text"
            ttc_text.id = 2000 + trk.id
            ttc_text.type = Marker.TEXT_VIEW_FACING
            ttc_text.action = Marker.ADD

            ttc_text.pose.position.x = x
            ttc_text.pose.position.y = y
            ttc_text.pose.position.z = 4.0

            ttc_text.scale.z = 0.8

            if ttc < 2.0:
                ttc_text.color.r = 1.0
                ttc_text.color.g = 0.0
                ttc_text.color.b = 0.0
            else:
                ttc_text.color.r = 1.0
                ttc_text.color.g = 1.0
                ttc_text.color.b = 0.0

            ttc_text.color.a = 1.0

            if ttc == float("inf"):
                ttc_text.text = "TTC: inf"
            else:
                ttc_text.text = f"TTC: {ttc:.2f}s"

            marker_array.markers.append(ttc_text)

        self.marker_pub.publish(marker_array)


def main():
    rclpy.init()
    node = GTTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()