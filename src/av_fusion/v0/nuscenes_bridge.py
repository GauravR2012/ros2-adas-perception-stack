#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from nuscenes.nuscenes import NuScenes


def quat_to_yaw(w, x, y, z):
    """Extract yaw from quaternion (nuScenes convention: w, x, y, z)."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class NuScenesBridge(Node):
    def __init__(self):
        super().__init__('nuscenes_bridge')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('dataset_path',     '')
        self.declare_parameter('version',          'v1.0-mini')
        self.declare_parameter('playback_rate_hz',  10.0)
        self.declare_parameter('scene_index',       0)

        dataset_path = self.get_parameter('dataset_path').get_parameter_value().string_value
        version      = self.get_parameter('version').get_parameter_value().string_value
        rate         = self.get_parameter('playback_rate_hz').get_parameter_value().double_value
        scene_index  = self.get_parameter('scene_index').get_parameter_value().integer_value

        if not dataset_path:
            self.get_logger().error(
                "dataset_path parameter is required. "
                "Pass with: --ros-args -p dataset_path:=/path/to/nuscenes")
            raise SystemExit

        # ── Load nuScenes ─────────────────────────────────────────────────────
        self.get_logger().info(f"Loading nuScenes {version} from {dataset_path} ...")
        self.nusc = NuScenes(version=version, dataroot=dataset_path, verbose=False)

        if scene_index >= len(self.nusc.scene):
            self.get_logger().error(
                f"scene_index {scene_index} out of range "
                f"(dataset has {len(self.nusc.scene)} scenes)")
            raise SystemExit

        self.scene        = self.nusc.scene[scene_index]
        self.sample_token = self.scene['first_sample_token']
        self.get_logger().info(
            f"Playing scene {scene_index}: '{self.scene['name']}' "
            f"({self.scene['nbr_samples']} samples at {rate} Hz)")

        # ── Publishers ────────────────────────────────────────────────────────
        self.pose_pub = self.create_publisher(PoseStamped, '/pose_measurement', 10)
        self.odom_pub = self.create_publisher(Odometry,    '/odom',             10)
        self.imu_pub  = self.create_publisher(Imu,         '/imu',              10)

        # ── State ─────────────────────────────────────────────────────────────
        self.prev_nuscenes_time = None   # microsecond timestamp
        self.prev_x             = None
        self.prev_y             = None

        # Previous quaternion — used to compute yaw rate for IMU
        self.prev_qw = None
        self.prev_qx = None
        self.prev_qy = None
        self.prev_qz = None

        # ── Playback timer ────────────────────────────────────────────────────
        self.timer = self.create_timer(1.0 / rate, self.timer_callback)

    # ── Timer callback ────────────────────────────────────────────────────────
    def timer_callback(self):
        if not self.sample_token:
            self.get_logger().info("🔁 Looping nuScenes scene...")

            # Reset to beginning of scene
            self.sample_token = self.scene['first_sample_token']

            # Reset previous state (VERY IMPORTANT)
            self.prev_nuscenes_time = None
            self.prev_x = None
            self.prev_y = None
            self.prev_qw = None
            self.prev_qx = None
            self.prev_qy = None
            self.prev_qz = None

            return

        # ── Fetch nuScenes data ───────────────────────────────────────────────
        sample   = self.nusc.get('sample',      self.sample_token)
        sd       = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose',    sd['ego_pose_token'])

        # nuScenes timestamp in microseconds — used for dt computation only
        ns_timestamp = sd['timestamp']

        x = ego_pose['translation'][0]
        y = ego_pose['translation'][1]

        # nuScenes quaternion convention: [w, x, y, z]
        q_w = ego_pose['rotation'][0]
        q_x = ego_pose['rotation'][1]
        q_y = ego_pose['rotation'][2]
        q_z = ego_pose['rotation'][3]

        # ── Compute dt from nuScenes timestamps ───────────────────────────────
        dt = 0.0
        if self.prev_nuscenes_time is not None:
            dt = (ns_timestamp - self.prev_nuscenes_time) / 1e6   # seconds

        # ── Compute forward velocity ──────────────────────────────────────────
        # Project displacement onto heading — gives signed forward velocity,
        # not scalar speed. Correct for a vehicle with non-holonomic motion.
        v = 0.0
        if dt > 0.0 and self.prev_x is not None:
            dx  = x - self.prev_x
            dy  = y - self.prev_y
            yaw = quat_to_yaw(q_w, q_x, q_y, q_z)
            v   = (dx * math.cos(yaw) + dy * math.sin(yaw)) / dt

        # ── Compute yaw rate from consecutive quaternions ─────────────────────
        # Derived from ego_pose rotation difference — no raw IMU needed.
        # The EKF uses this as yaw_rate_ in predict() to steer the prediction
        # correctly between pose corrections, dramatically reducing NIS.
        yaw_rate = 0.0
        if dt > 0.0 and self.prev_qw is not None:
            curr_yaw = quat_to_yaw(q_w, q_x, q_y, q_z)
            prev_yaw = quat_to_yaw(
                self.prev_qw, self.prev_qx, self.prev_qy, self.prev_qz)
            # Normalise angle difference to (-π, π] before dividing by dt
            dyaw = math.atan2(
                math.sin(curr_yaw - prev_yaw),
                math.cos(curr_yaw - prev_yaw))
            yaw_rate = dyaw / dt

        # ── ROS wall clock stamp ──────────────────────────────────────────────
        # nuScenes timestamps are 2018 unix epoch. EKF computes
        # dt = now() - last_time using ROS clock. Using nuScenes stamps
        # would give dt ~8 years, triggering the dt>1.0 guard every message.
        ros_stamp = self.get_clock().now().to_msg()

        # ── Publish IMU ───────────────────────────────────────────────────────
        imu_msg = Imu()
        imu_msg.header.stamp    = ros_stamp
        imu_msg.header.frame_id = 'imu_link'
        imu_msg.angular_velocity.z = yaw_rate
        # Covariance: diagonal, yaw-axis only
        imu_msg.angular_velocity_covariance[8] = 0.01 * 0.01
        # Orientation not used by EKF — leave as zero
        imu_msg.orientation_covariance[0] = -1.0   # signals "not provided"
        self.imu_pub.publish(imu_msg)

        # ── Publish PoseStamped ───────────────────────────────────────────────
        pose_msg = PoseStamped()
        pose_msg.header.stamp    = ros_stamp
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = 0.0
        # ROS2 quaternion order: x, y, z, w — nuScenes is w, x, y, z
        pose_msg.pose.orientation.w = q_w
        pose_msg.pose.orientation.x = q_x
        pose_msg.pose.orientation.y = q_y
        pose_msg.pose.orientation.z = q_z
        self.pose_pub.publish(pose_msg)

        # ── Publish Odometry ──────────────────────────────────────────────────
        odom_msg = Odometry()
        odom_msg.header.stamp    = ros_stamp
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id  = 'base_link'
        odom_msg.pose.pose.position.x    = x
        odom_msg.pose.pose.position.y    = y
        odom_msg.pose.pose.position.z    = 0.0
        odom_msg.pose.pose.orientation.w = q_w
        odom_msg.pose.pose.orientation.x = q_x
        odom_msg.pose.pose.orientation.y = q_y
        odom_msg.pose.pose.orientation.z = q_z
        odom_msg.twist.twist.linear.x    = v
        odom_msg.twist.twist.linear.y    = 0.0
        odom_msg.twist.twist.linear.z    = 0.0
        self.odom_pub.publish(odom_msg)

        self.get_logger().debug(
            f"Sample {self.sample_token[:8]} | "
            f"x={x:.2f} y={y:.2f} v={v:.3f} m/s yaw_rate={yaw_rate:.4f} rad/s")

        # ── Update state for next iteration ───────────────────────────────────
        self.prev_nuscenes_time = ns_timestamp
        self.prev_x             = x
        self.prev_y             = y
        self.prev_qw            = q_w
        self.prev_qx            = q_x
        self.prev_qy            = q_y
        self.prev_qz            = q_z
        self.sample_token       = sample['next']


# ── Entry point ───────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = NuScenesBridge()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    
    main()
