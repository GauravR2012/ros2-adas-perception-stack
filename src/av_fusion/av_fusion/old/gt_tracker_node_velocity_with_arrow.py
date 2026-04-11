import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped

import numpy as np
from scipy.optimize import linear_sum_assignment

import tf2_ros
from tf2_geometry_msgs import do_transform_point
from rclpy.time import Time
from rclpy.duration import Duration
import math


# =========================================================
# Kalman Filter Tracker (Constant Velocity Model)
# =========================================================
class KalmanTracker:

    count = 0

    def __init__(self, x, y):
        self.x = np.array([[x], [y], [0.0], [0.0]])
        self.P = np.eye(4) * 10.0

        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]])

        self.R = np.eye(2)
        self.Q = np.eye(4) * 0.05

        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        self.missed = 0

    def predict(self, dt):
        F = np.array([[1,0,dt,0],
                      [0,1,0,dt],
                      [0,0,1,0],
                      [0,0,0,1]])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        z = np.array([[z[0]],[z[1]]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.missed = 0

    def mahalanobis(self, z):
        z = np.array([[z[0]],[z[1]]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return float(y.T @ np.linalg.inv(S) @ y)


# =========================================================
# Tracker Node
# =========================================================
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

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.trackers = []
        self.max_missed = 5
        self.last_time = None
        self.lane_width = 3.5

        self.get_logger().info("Tracker + Velocity + TTC Started")

    # ------------------------------------------------------
    def callback(self, msg):

        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_time is None:
            self.last_time = current_time
            for det in msg.detections:
                self.trackers.append(
                    KalmanTracker(
                        det.bbox.center.position.x,
                        det.bbox.center.position.y
                    )
                )
            return

        dt = current_time - self.last_time
        self.last_time = current_time

        # ---------------- Detections ----------------
        detections = []
        for det in msg.detections:
            detections.append([
                det.bbox.center.position.x,
                det.bbox.center.position.y
            ])
        detections = np.array(detections)

        # ---------------- Predict ----------------
        for trk in self.trackers:
            trk.predict(dt)

        # ---------------- Associate ----------------
        if len(detections) > 0 and len(self.trackers) > 0:

            cost_matrix = np.zeros((len(detections), len(self.trackers)))

            for i, det in enumerate(detections):
                for j, trk in enumerate(self.trackers):
                    cost_matrix[i, j] = trk.mahalanobis(det)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assigned_tracks = set()
            assigned_dets = set()

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 25.0:
                    self.trackers[c].update(detections[r])
                    assigned_tracks.add(c)
                    assigned_dets.add(r)

            for i, trk in enumerate(self.trackers):
                if i not in assigned_tracks:
                    trk.missed += 1

            for i, det in enumerate(detections):
                if i not in assigned_dets:
                    self.trackers.append(KalmanTracker(det[0], det[1]))

        self.trackers = [t for t in self.trackers if t.missed < self.max_missed]

        # ---------------- TF ----------------
        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link",
                "map",
                Time(),
                timeout=Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warn(f"TF not ready: {e}")
            return

        marker_array = MarkerArray()

        for trk in self.trackers:

            x_map = trk.x[0,0]
            y_map = trk.x[1,0]
            vx_map = trk.x[2,0]
            vy_map = trk.x[3,0]

            # -------- Position to Ego Frame --------
            pt = PointStamped()
            pt.header.frame_id = "map"
            pt.point.x = x_map
            pt.point.y = y_map
            pt.point.z = 0.0

            ego_pt = do_transform_point(pt, transform)

            long_dist = ego_pt.point.x
            lat_dist = ego_pt.point.y

            # -------- Velocity Rotation --------
            q = transform.transform.rotation
            sin_yaw = 2.0 * (q.w * q.z + q.x * q.y)
            cos_yaw = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
            yaw = math.atan2(sin_yaw, cos_yaw)

            R = np.array([
                [math.cos(yaw), math.sin(yaw)],
                [-math.sin(yaw), math.cos(yaw)]
            ])

            vel_ego = R @ np.array([[vx_map],[vy_map]])
            v_long = vel_ego[0,0]

            # -------- TTC --------
            in_lane = abs(lat_dist) < (self.lane_width / 2)

            if long_dist > 0 and v_long < 0 and in_lane:
                ttc = long_dist / (-v_long)
            else:
                ttc = float("inf")

            # =====================================================
            # RED TRACK BOX
            # =====================================================
            box = Marker()
            box.header.frame_id = "map"
            box.header.stamp = msg.header.stamp
            box.ns = "tracked"
            box.id = trk.id
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose.position.x = x_map
            box.pose.position.y = y_map
            box.pose.position.z = 0.0
            box.scale.x = 2.0
            box.scale.y = 1.0
            box.scale.z = 1.5
            box.color.r = 1.0
            box.color.g = 0.0
            box.color.b = 0.0
            box.color.a = 0.8

            marker_array.markers.append(box)

            # =====================================================
            # BLUE VELOCITY ARROW
            # =====================================================
            arrow = Marker()
            arrow.header.frame_id = "map"
            arrow.header.stamp = msg.header.stamp
            arrow.ns = "velocity"
            arrow.id = 10000 + trk.id
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD

            arrow.scale.x = 0.3
            arrow.scale.y = 0.6
            arrow.scale.z = 0.6

            arrow.color.r = 0.0
            arrow.color.g = 0.0
            arrow.color.b = 1.0
            arrow.color.a = 1.0

            start = Point()
            start.x = x_map
            start.y = y_map
            start.z = 0.0

            end = Point()
            end.x = x_map + vx_map
            end.y = y_map + vy_map
            end.z = 0.0

            arrow.points = [start, end]

            marker_array.markers.append(arrow)

            # =====================================================
            # TTC TEXT
            # =====================================================
            ttc_marker = Marker()
            ttc_marker.header.frame_id = "map"
            ttc_marker.header.stamp = msg.header.stamp
            ttc_marker.ns = "ttc"
            ttc_marker.id = 20000 + trk.id
            ttc_marker.type = Marker.TEXT_VIEW_FACING
            ttc_marker.action = Marker.ADD

            ttc_marker.pose.position.x = x_map
            ttc_marker.pose.position.y = y_map
            ttc_marker.pose.position.z = 3.0

            ttc_marker.scale.z = 0.9
            ttc_marker.color.a = 1.0

            if ttc == float("inf"):
                ttc_marker.text = "TTC: inf"
                ttc_marker.color.r = 0.0
                ttc_marker.color.g = 1.0
                ttc_marker.color.b = 0.0
            else:
                ttc_marker.text = f"TTC: {ttc:.2f}s"

                if ttc < 1.5:
                    ttc_marker.color.r = 1.0
                    ttc_marker.color.g = 0.0
                    ttc_marker.color.b = 0.0
                elif ttc < 3.0:
                    ttc_marker.color.r = 1.0
                    ttc_marker.color.g = 0.5
                    ttc_marker.color.b = 0.0
                else:
                    ttc_marker.color.r = 0.0
                    ttc_marker.color.g = 1.0
                    ttc_marker.color.b = 0.0

            marker_array.markers.append(ttc_marker)

        self.marker_pub.publish(marker_array)


# =========================================================
def main():
    rclpy.init()
    node = GTTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()