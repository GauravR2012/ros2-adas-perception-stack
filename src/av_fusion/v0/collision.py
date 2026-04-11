import rclpy
from rclpy.node import Node

from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
import time


class Track:
    def __init__(self, track_id, position, timestamp):
        self.id = track_id
        self.position = position
        self.prev_position = position
        self.last_time = timestamp
        self.velocity = np.array([0.0, 0.0, 0.0])


class CollisionEstimator(Node):

    def __init__(self):
        super().__init__("collision_estimator")

        self.sub = self.create_subscription(
            Detection3DArray,
            "/detections/boxes_3d",
            self.callback,
            10
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/tracking/markers",
            10
        )

        self.tracks = []
        self.next_id = 0

        self.get_logger().info("🚗 Tracker + Velocity + TTC (RViz Ready)")

    # ---------------------------------------------------------
    def extract_objects(self, msg):
        objs = []

        for det in msg.detections:
            bbox = det.bbox
            x = bbox.center.position.x
            y = bbox.center.position.y
            z = bbox.center.position.z

            objs.append(np.array([x, y, z]))

        return objs

    # ---------------------------------------------------------
    def match_tracks(self, detections):
        current_time = time.time()

        assigned_tracks = set()

        for det in detections:

            best_track = None
            best_dist = 1e9

            for track in self.tracks:
                dist = np.linalg.norm(det[:2] - track.position[:2])

                if dist < best_dist and dist < 3.0:
                    best_dist = dist
                    best_track = track

            if best_track is not None:
                # update track
                dt = current_time - best_track.last_time

                if dt > 0:
                    best_track.velocity = (det - best_track.position) / dt

                best_track.prev_position = best_track.position
                best_track.position = det
                best_track.last_time = current_time

                assigned_tracks.add(best_track.id)

            else:
                # create new track
                new_track = Track(self.next_id, det, current_time)
                self.tracks.append(new_track)
                self.next_id += 1

        # remove stale tracks
        self.tracks = [
            t for t in self.tracks if (current_time - t.last_time) < 1.0
        ]

    # ---------------------------------------------------------
    def publish_markers(self):
        marker_array = MarkerArray()
        marker_id = 0

        for track in self.tracks:

            pos = track.position
            vel = track.velocity

            # -------- ARROW (velocity) --------
            arrow = Marker()
            arrow.header.frame_id = "map"
            arrow.header.stamp = self.get_clock().now().to_msg()
            arrow.id = marker_id
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD

            arrow.scale.x = 0.2
            arrow.scale.y = 0.4
            arrow.scale.z = 0.4

            arrow.color.r = 0.0
            arrow.color.g = 0.0
            arrow.color.b = 1.0
            arrow.color.a = 1.0

            arrow.points = []

            start = pos
            end = pos + vel

            from geometry_msgs.msg import Point
            p1 = Point(x=float(start[0]), y=float(start[1]), z=float(start[2]))
            p2 = Point(x=float(end[0]), y=float(end[1]), z=float(end[2]))

            arrow.points.append(p1)
            arrow.points.append(p2)

            marker_array.markers.append(arrow)
            marker_id += 1

            # -------- TEXT (ID) --------
            text = Marker()
            text.header.frame_id = "map"
            text.header.stamp = self.get_clock().now().to_msg()
            text.id = marker_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD

            text.pose.position.x = float(pos[0])
            text.pose.position.y = float(pos[1])
            text.pose.position.z = float(pos[2] + 1.5)

            text.scale.z = 1.0

            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0

            text.text = f"ID {track.id}"

            marker_array.markers.append(text)
            marker_id += 1

        self.marker_pub.publish(marker_array)

    # ---------------------------------------------------------
    def compute_ttc(self):

        for track in self.tracks:

            x = track.position[0]
            vx = track.velocity[0]

            if abs(track.position[1]) > 10:
                continue

            if vx >= 0:
                continue

            if abs(vx) < 0.05:
                continue

            ttc = abs(x / vx)

            self.get_logger().info(
                f"ID {track.id} | Dist: {x:.2f} | Vel: {vx:.2f} | TTC: {ttc:.2f}"
            )

            if ttc < 5.0:
                self.get_logger().warn("🚨 BRAKE APPLIED")

    # ---------------------------------------------------------
    def callback(self, msg):

        detections = self.extract_objects(msg)

        self.match_tracks(detections)

        self.compute_ttc()

        self.publish_markers()


# ---------------------------------------------------------
def main():
    rclpy.init()
    node = CollisionEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()