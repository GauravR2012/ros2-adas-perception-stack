import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from scipy.optimize import linear_sum_assignment


class Track:

    def __init__(self, id, position, size=None):
        self.id = id
        self.position = np.array(position)
        self.size = np.array(size) if size is not None else np.array([2.0, 1.0, 1.5])
        self.velocity = np.zeros(3)
        self.last_update = None
        self.missed = 0


class StableTracker(Node):

    def __init__(self):
        super().__init__('stable_tracker')

        # Subscribe to detections topic
        self.sub = self.create_subscription(
            Detection3DArray,
            '/detections/boxes_3d',
            self.callback,
            10
        )

        # Publish visualization markers
        self.pub = self.create_publisher(
            MarkerArray,
            '/tracking/markers',
            10
        )

        # Publish tracked objects message
        self.tracked_objects_pub = self.create_publisher(
            Detection3DArray,
            '/tracked_objects',
            10
        )

        self.tracks = []
        self.next_id = 0

        # PARAMETERS
        self.max_distance = 2.5   # matching threshold (meters)
        self.max_missed = 5      # track deletion
        self.alpha = 0.6         # velocity smoothing

        self.get_logger().info("🚀 Stable Tracker (HUNGARIAN) Started")

    def callback(self, msg):
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        detections = []
        sizes = []

        for det in msg.detections:
            p = det.bbox.center.position
            s = det.bbox.size
            detections.append(np.array([p.x, p.y, p.z]))
            sizes.append(np.array([s.x, s.y, s.z]))

        detections = np.array(detections)
        sizes = np.array(sizes)

        if len(detections) == 0:
            # Increment missed for existing tracks
            for track in self.tracks:
                track.missed += 1
            self.tracks = [t for t in self.tracks if t.missed < self.max_missed]
            # Publish empty tracked objects message
            tracked_msg = Detection3DArray()
            tracked_msg.header = msg.header
            self.tracked_objects_pub.publish(tracked_msg)
            return

        self.update_tracks(detections, sizes, stamp_sec)
        self.publish_markers(msg.header)
        self.publish_tracked_objects(msg.header)

    def update_tracks(self, detections, sizes, stamp_sec):
        num_tracks = len(self.tracks)
        num_dets = len(detections)

        if num_tracks == 0:
            for i in range(num_dets):
                new_track = Track(self.next_id, detections[i], sizes[i])
                new_track.last_update = stamp_sec
                self.tracks.append(new_track)
                self.next_id += 1
            return

        # =========================
        # DATA ASSOCIATION (HUNGARIAN)
        # =========================
        cost_matrix = np.zeros((num_tracks, num_dets))
        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                cost_matrix[t_idx, d_idx] = np.linalg.norm(det - track.position)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_dets = set()
        assigned_tracks = set()

        for t_idx, d_idx in zip(row_ind, col_ind):
            dist = cost_matrix[t_idx, d_idx]
            if dist < self.max_distance:
                track = self.tracks[t_idx]
                det = detections[d_idx]
                size = sizes[d_idx]

                if track.last_update is not None:
                    dt = stamp_sec - track.last_update
                    if dt > 0:
                        measured_vel = (det - track.position) / dt
                        # Smooth velocity
                        track.velocity = (
                            self.alpha * measured_vel +
                            (1 - self.alpha) * track.velocity
                        )

                track.position = det
                track.size = size
                track.last_update = stamp_sec
                track.missed = 0

                assigned_dets.add(d_idx)
                assigned_tracks.add(t_idx)

        # Handle unassigned tracks
        for t_idx in range(num_tracks):
            if t_idx not in assigned_tracks:
                self.tracks[t_idx].missed += 1

        # Create new tracks for unassigned detections
        for d_idx in range(num_dets):
            if d_idx not in assigned_dets:
                new_track = Track(self.next_id, detections[d_idx], sizes[d_idx])
                new_track.last_update = stamp_sec
                self.tracks.append(new_track)
                self.next_id += 1

        # Delete lost tracks
        self.tracks = [t for t in self.tracks if t.missed < self.max_missed]

    def publish_tracked_objects(self, header):
        tracked_msg = Detection3DArray()
        tracked_msg.header = header

        for track in self.tracks:
            if track.missed > 0:
                continue

            det = Detection3D()
            det.header = header

            # Position
            det.bbox.center.position.x = float(track.position[0])
            det.bbox.center.position.y = float(track.position[1])
            det.bbox.center.position.z = float(track.position[2])

            # Dimension propagation
            det.bbox.size.x = float(track.size[0])
            det.bbox.size.y = float(track.size[1])
            det.bbox.size.z = float(track.size[2])

            # Encode Tracking ID and Velocity Vector in hypothesis field
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(track.id)
            hyp.hypothesis.score = 1.0

            hyp.pose.pose.position.x = float(track.velocity[0])
            hyp.pose.pose.position.y = float(track.velocity[1])
            hyp.pose.pose.position.z = float(track.velocity[2])

            det.results.append(hyp)
            tracked_msg.detections.append(det)

        self.tracked_objects_pub.publish(tracked_msg)

    def publish_markers(self, header):
        marker_array = MarkerArray()

        # ── DELETEALL at the start of every publish cycle ──
        # Wipes all stale markers from deleted/re-IDed tracks so the
        # red boxes don't accumulate indefinitely in RViz.
        delete_all = Marker()
        delete_all.header = header
        delete_all.ns = "tracks"
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for track in self.tracks:
            if track.missed > 0:
                continue

            # =========================
            # BOX MARKER (Actual scale)
            # =========================
            marker = Marker()
            marker.header = header
            marker.ns = "tracks"
            marker.id = track.id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = float(track.position[0])
            marker.pose.position.y = float(track.position[1])
            marker.pose.position.z = float(track.position[2])

            marker.scale.x = float(track.size[0])
            marker.scale.y = float(track.size[1])
            marker.scale.z = float(track.size[2])

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8

            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 600000000

            marker_array.markers.append(marker)

            # =========================
            # VELOCITY ARROW
            # =========================
            arrow = Marker()
            arrow.header = header
            arrow.ns = "velocity"
            arrow.id = track.id + 1000
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD

            arrow.lifetime.sec = 0
            arrow.lifetime.nanosec = 600000000

            arrow.pose.position.x = float(track.position[0])
            arrow.pose.position.y = float(track.position[1])
            arrow.pose.position.z = float(track.position[2])

            speed = np.linalg.norm(track.velocity)
            arrow.scale.x = float(speed * 2.0) if speed > 0.05 else 0.01
            arrow.scale.y = 0.3
            arrow.scale.z = 0.3

            arrow.color.r = 0.0
            arrow.color.g = 0.0
            arrow.color.b = 1.0
            arrow.color.a = 1.0

            # Orientation of arrow towards velocity vector
            if speed > 0.05:
                yaw = np.arctan2(track.velocity[1], track.velocity[0])
                arrow.pose.orientation.z = float(np.sin(yaw / 2.0))
                arrow.pose.orientation.w = float(np.cos(yaw / 2.0))

            marker_array.markers.append(arrow)

        self.pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = StableTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()