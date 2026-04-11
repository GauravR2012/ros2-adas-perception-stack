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
import math


# =========================================
# Kalman Tracker
# =========================================
class KalmanTracker:

    count = 0

    def __init__(self, x, y):
        self.x = np.array([[x], [y], [0.0], [0.0]])
        self.P = np.eye(4) * 10.0

        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]])

        self.R = np.eye(2) * 1.0
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

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.trackers = []
        self.max_missed = 5
        self.dist_threshold = 3.0
        self.last_time = None

        # Lane width (meters)
        self.lane_width = 3.5

        self.get_logger().info("Tracker + Lane Filtering + TTC Started")

    # ---------------------------------------------------
    def callback(self, msg):

        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_time is None:
            self.last_time = current_time
            return

        dt = current_time - self.last_time
        self.last_time = current_time

        # ---------------- DETECTIONS ----------------
        detections = []
        for det in msg.detections:
            detections.append([
                det.bbox.center.position.x,
                det.bbox.center.position.y
            ])

        detections = np.array(detections)

        # ---------------- PREDICT ----------------
        predictions = []
        for trk in self.trackers:
            trk.predict(dt)
            predictions.append([trk.x[0,0], trk.x[1,0]])

        predictions = np.array(predictions) if len(predictions)>0 else np.empty((0,2))

        # ---------------- ASSOCIATION ----------------
        if len(detections)>0 and len(predictions)>0:

            cost_matrix = np.linalg.norm(
                detections[:,None,:] - predictions[None,:,:],
                axis=2
            )

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assigned_tracks = set()
            assigned_dets = set()

            for r,c in zip(row_ind, col_ind):
                if cost_matrix[r,c] < self.dist_threshold:
                    self.trackers[c].update(detections[r])
                    assigned_tracks.add(c)
                    assigned_dets.add(r)

            for i,trk in enumerate(self.trackers):
                if i not in assigned_tracks:
                    trk.missed += 1

            for i,det in enumerate(detections):
                if i not in assigned_dets:
                    self.trackers.append(KalmanTracker(det[0], det[1]))

        else:
            for det in detections:
                self.trackers.append(KalmanTracker(det[0], det[1]))

        self.trackers = [t for t in self.trackers if t.missed < self.max_missed]

        # ---------------- TF LOOKUP ----------------
        try:
            transform = self.tf_buffer.lookup_transform(
                "base_link",
                msg.header.frame_id,
                Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=0.1)
            )
        except:
            return

        # Extract ego yaw from quaternion
        q = transform.transform.rotation
        sin_yaw = 2.0 * (q.w * q.z + q.x * q.y)
        cos_yaw = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        ego_yaw = math.atan2(sin_yaw, cos_yaw)

        R = np.array([
            [math.cos(ego_yaw), math.sin(ego_yaw)],
            [-math.sin(ego_yaw), math.cos(ego_yaw)]
        ])

        marker_array = MarkerArray()

        for trk in self.trackers:

            x = trk.x[0,0]
            y = trk.x[1,0]
            vx_map = trk.x[2,0]
            vy_map = trk.x[3,0]

            # Transform velocity to ego frame
            vel_ego = R @ np.array([[vx_map],[vy_map]])
            v_long = vel_ego[0,0]

            # Transform position to ego frame
            pt = PointStamped()
            pt.header = msg.header
            pt.point.x = x
            pt.point.y = y
            pt.point.z = 0.0

            ego_pt = do_transform_point(pt, transform)
            long_dist = ego_pt.point.x
            lat_dist = ego_pt.point.y

            # -------- LANE FILTERING --------
            in_lane = abs(lat_dist) < (self.lane_width / 2)

            # -------- TTC --------
            if long_dist > 0 and v_long < 0 and in_lane:
                ttc = long_dist / (-v_long)
            else:
                ttc = float("inf")

            # ---------------- BOX ----------------
            box = Marker()
            box.header = msg.header
            box.ns = "tracked"
            box.id = trk.id
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose.position.x = x
            box.pose.position.y = y
            box.pose.position.z = 0.0
            box.scale.x = 2.0
            box.scale.y = 1.0
            box.scale.z = 1.5

            if in_lane:
                box.color.r = 1.0
                box.color.g = 0.0
                box.color.b = 0.0
                box.color.a = 0.8
            else:
                # Fade out-of-lane objects
                box.color.r = 0.5
                box.color.g = 0.5
                box.color.b = 0.5
                box.color.a = 0.2

            marker_array.markers.append(box)

            # ---------------- TTC TEXT ----------------
            ttc_marker = Marker()
            ttc_marker.header = msg.header
            ttc_marker.ns = "ttc"
            ttc_marker.id = 1000 + trk.id
            ttc_marker.type = Marker.TEXT_VIEW_FACING
            ttc_marker.action = Marker.ADD
            ttc_marker.pose.position.x = x
            ttc_marker.pose.position.y = y
            ttc_marker.pose.position.z = 4.0
            ttc_marker.scale.z = 0.9

            if ttc == float("inf"):
                ttc_marker.color.r = 0.0
                ttc_marker.color.g = 1.0
                ttc_marker.color.b = 0.0
            elif ttc < 1.5:
                ttc_marker.color.r = 1.0
                ttc_marker.color.g = 0.0
                ttc_marker.color.b = 0.0
            elif ttc < 3.0:
                ttc_marker.color.r = 1.0
                ttc_marker.color.g = 0.5
                ttc_marker.color.b = 0.0
            elif ttc < 6.0:
                ttc_marker.color.r = 1.0
                ttc_marker.color.g = 1.0
                ttc_marker.color.b = 0.0
            else:
                ttc_marker.color.r = 0.0
                ttc_marker.color.g = 1.0
                ttc_marker.color.b = 0.0

            ttc_marker.color.a = 1.0
            ttc_marker.text = "TTC: inf" if ttc == float("inf") else f"TTC: {ttc:.2f}s"

            marker_array.markers.append(ttc_marker)

        self.marker_pub.publish(marker_array)


def main():
    rclpy.init()
    node = GTTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()