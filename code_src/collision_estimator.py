import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import time
import math


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
            "/collision_markers",
            10
        )

        # -----------------------------
        # tracking memory
        # -----------------------------
        self.prev_tracks = {}
        self.next_id = 0
        self.prev_time = None

        # -----------------------------
        # kalman-style smoothing
        # -----------------------------
        self.alpha = 0.75   # keep earlier preference: stable smoothing

        # -----------------------------
        # metrics
        # -----------------------------
        self.total_frames = 0
        self.total_objects = 0
        self.brake_events = 0
        self.min_ttc_seen = 9999.0

        self.get_logger().info("🚗 Collision Estimator + Tracking + Metrics Started")

    # ==========================================================
    # nearest-neighbor tracker
    # ==========================================================
    def assign_id(self, x, y):
        best_id = None
        best_dist = 9999

        for obj_id, track in self.prev_tracks.items():
            dist = math.sqrt((x - track["x"]) ** 2 + (y - track["y"]) ** 2)
            if dist < best_dist and dist < 2.5:
                best_dist = dist
                best_id = obj_id

        if best_id is None:
            best_id = self.next_id
            self.next_id += 1

        return best_id

    # ==========================================================
    # main callback
    # ==========================================================
    def callback(self, msg):
        current_time = time.time()

        if self.prev_time is None:
            self.prev_time = current_time
            return

        dt = current_time - self.prev_time
        dt = max(dt, 0.05)

        self.total_frames += 1
        self.total_objects += len(msg.detections)

        marker_array = MarkerArray()
        new_tracks = {}

        brake_triggered = False
        frame_min_ttc = 9999.0

        for det in msg.detections:
            x = det.bbox.center.position.x
            y = det.bbox.center.position.y
            z = det.bbox.center.position.z

            obj_id = self.assign_id(x, y)

            prev = self.prev_tracks.get(
                obj_id,
                {"x": x, "y": y, "vx": 0.0, "vy": 0.0}
            )

            raw_vx = (x - prev["x"]) / dt
            raw_vy = (y - prev["y"]) / dt

            # =====================================
            # Kalman-style EMA smoothing
            # =====================================
            vx = self.alpha * prev["vx"] + (1 - self.alpha) * raw_vx
            vy = self.alpha * prev["vy"] + (1 - self.alpha) * raw_vy

            speed = math.sqrt(vx * vx + vy * vy)

            distance = math.sqrt(x * x + y * y)

            # use forward relative speed
            rel_speed = -vx

            if rel_speed > 0.1:
                ttc = distance / rel_speed
            else:
                ttc = 9999.0

            frame_min_ttc = min(frame_min_ttc, ttc)
            self.min_ttc_seen = min(self.min_ttc_seen, ttc)

            # =====================================
            # danger logic
            # =====================================
            danger = ttc < 2.0 and x > 0 and abs(y) < 4.0

            if danger:
                brake_triggered = True

            # =====================================
            # save track
            # =====================================
            new_tracks[obj_id] = {
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy
            }

            # =====================================
            # TEXT MARKER
            # =====================================
            text = Marker()
            text.header = msg.header
            text.ns = "collision_text"
            text.id = obj_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD

            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = z + 2.0

            text.scale.z = 0.6
            text.color.a = 1.0

            if danger:
                text.color.r = 1.0
                text.color.g = 0.0
                text.color.b = 0.0
                text.text = f"ID {obj_id}\nDANGER\nTTC {ttc:.1f}s"
            else:
                text.color.r = 0.0
                text.color.g = 1.0
                text.color.b = 0.0
                text.text = f"ID {obj_id}\nSAFE\nTTC {ttc:.1f}s"

            marker_array.markers.append(text)

            # =====================================
            # VELOCITY ARROW
            # =====================================
            arrow = Marker()
            arrow.header = msg.header
            arrow.ns = "velocity"
            arrow.id = 1000 + obj_id
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD

            display_vx = max(min(vx, 3.0), -3.0)
            display_vy = max(min(vy, 3.0), -3.0)

            p1 = Point()
            p2 = Point()

            p1.x = x
            p1.y = y
            p1.z = z + 1.0

            p2.x = x + display_vx * 0.8
            p2.y = y + display_vy * 0.8
            p2.z = z + 1.0

            arrow.points = [p1, p2]

            arrow.scale.x = 0.08
            arrow.scale.y = 0.18
            arrow.scale.z = 0.18

            arrow.color.a = 1.0
            arrow.color.r = 0.0
            arrow.color.g = 0.0
            arrow.color.b = 1.0

            marker_array.markers.append(arrow)

        # =====================================
        # METRICS + BRAKE
        # =====================================
        if brake_triggered:
            self.brake_events += 1
            self.get_logger().warn(
                f"🛑 BRAKE | TTC {frame_min_ttc:.2f}s"
            )
        else:
            self.get_logger().info(
                f"✅ SAFE | TTC {frame_min_ttc:.2f}s"
            )

        avg_objects = self.total_objects / max(self.total_frames, 1)

        self.get_logger().info(
            f"📊 Frames={self.total_frames} | "
            f"AvgObj={avg_objects:.1f} | "
            f"Brakes={self.brake_events} | "
            f"MinTTC={self.min_ttc_seen:.2f}"
        )

        self.marker_pub.publish(marker_array)

        self.prev_tracks = new_tracks
        self.prev_time = current_time


def main():
    rclpy.init()
    node = CollisionEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()