import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import math


class CollisionEstimator(Node):
    def __init__(self):
        super().__init__("collision_estimator")

        self.sub = self.create_subscription(
            Detection3DArray,
            "/tracked_objects",
            self.callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/ekf/odom",
            self.odom_callback,
            10
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/collision_markers",
            10
        )

        # Ego state memory
        self.ego_pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}
        self.ego_vx = 0.0

        # metrics
        self.total_frames = 0
        self.total_objects = 0
        self.brake_events = 0
        self.min_ttc_seen = 9999.0

        self.get_logger().info("🚗 Collision Estimator Started (Centralized Tracker + EKF Odom)")

    def odom_callback(self, msg):
        self.ego_vx = msg.twist.twist.linear.x
        
        # Quaternion to Yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        self.ego_pose = {
            "x": msg.pose.pose.position.x,
            "y": msg.pose.pose.position.y,
            "yaw": yaw
        }

    def callback(self, msg):
        self.total_frames += 1
        self.total_objects += len(msg.detections)

        marker_array = MarkerArray()

        brake_triggered = False
        frame_min_ttc = 9999.0

        ego_x = self.ego_pose["x"]
        ego_y = self.ego_pose["y"]
        ego_yaw = self.ego_pose["yaw"]
        ego_vx = self.ego_vx

        is_map_frame = (msg.header.frame_id == "map")

        for det in msg.detections:
            ox = det.bbox.center.position.x
            oy = det.bbox.center.position.y
            oz = det.bbox.center.position.z

            # Extract Tracking ID and Velocity
            if len(det.results) > 0:
                obj_id = int(det.results[0].hypothesis.class_id)
                ovx = det.results[0].pose.pose.position.x
                ovy = det.results[0].pose.pose.position.y
            else:
                obj_id = 999
                ovx = 0.0
                ovy = 0.0

            # Transform absolute map coordinates to ego relative base_link frame
            if is_map_frame:
                dx = ox - ego_x
                dy = oy - ego_y
                x = dx * math.cos(ego_yaw) + dy * math.sin(ego_yaw)
                y = -dx * math.sin(ego_yaw) + dy * math.cos(ego_yaw)

                # Absolute velocity rotated to base_link
                vx_base = ovx * math.cos(ego_yaw) + ovy * math.sin(ego_yaw)
                vy_base = -ovx * math.sin(ego_yaw) + ovy * math.cos(ego_yaw)

                # True relative velocity by subtracting EKF forward velocity
                rel_vx = vx_base - ego_vx
                rel_vy = vy_base
            else:
                # Already in sensor/ego relative frame
                x = ox
                y = oy
                rel_vx = ovx
                rel_vy = ovy

            distance = math.sqrt(x * x + y * y)

            # Radial relative velocity: dot(pos, rel_vel) / |pos|
            # Approaches: rel_speed > 0
            rel_speed = - (x * rel_vx + y * rel_vy) / (distance + 1e-6)

            if rel_speed > 0.1:
                ttc = distance / rel_speed
            else:
                ttc = 9999.0

            frame_min_ttc = min(frame_min_ttc, ttc)
            self.min_ttc_seen = min(self.min_ttc_seen, ttc)

            # Danger logic (object inside ego path and approaching fast)
            danger = ttc < 2.0 and x > 0 and abs(y) < 4.0

            if danger:
                brake_triggered = True

            # =====================================
            # TEXT MARKER
            # =====================================
            text = Marker()
            text.header = msg.header
            text.ns = "collision_text"
            text.id = obj_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD

            # Keep text marker relative to base_link if we did the transform,
            # or in the incoming sensor frame otherwise.
            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = oz + 2.0

            # Override frame to base_link if we transformed to base_link
            if is_map_frame:
                text.header.frame_id = "base_link"

            text.scale.z = 0.6
            text.color.a = 1.0

            text.lifetime.sec = 0
            text.lifetime.nanosec = 600000000

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

            arrow.lifetime.sec = 0
            arrow.lifetime.nanosec = 600000000

            if is_map_frame:
                arrow.header.frame_id = "base_link"

            p1 = Point()
            p2 = Point()

            p1.x = x
            p1.y = y
            p1.z = oz + 1.0

            # Display arrow for object absolute velocity in ego frame (not relative)
            display_vx = vx_base if is_map_frame else rel_vx
            display_vy = vy_base if is_map_frame else rel_vy
            
            p2.x = x + display_vx * 0.8
            p2.y = y + display_vy * 0.8
            p2.z = oz + 1.0

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


def main():
    rclpy.init()
    node = CollisionEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()