#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import Header

from nuscenes.nuscenes import NuScenes

import struct
import os


def quat_to_yaw(w, x, y, z):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class NuScenesBridge(Node):
    def __init__(self):
        super().__init__('nuscenes_bridge')

        # PARAMETERS
        self.declare_parameter('dataset_path', '')
        self.declare_parameter('version', 'v1.0-mini')
        self.declare_parameter('playback_rate_hz', 10.0)
        self.declare_parameter('scene_index', 0)

        dataset_path = self.get_parameter('dataset_path').value
        version = self.get_parameter('version').value
        rate = self.get_parameter('playback_rate_hz').value
        scene_index = self.get_parameter('scene_index').value

        if not dataset_path:
            self.get_logger().error("dataset_path required")
            raise SystemExit

        self.nusc = NuScenes(version=version, dataroot=dataset_path, verbose=False)

        self.scene = self.nusc.scene[scene_index]
        self.sample_token = self.scene['first_sample_token']

        self.get_logger().info(
            f"Scene {scene_index}: {self.scene['name']}")

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/pose_measurement', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu', 10)
        self.pc_pub = self.create_publisher(PointCloud2, '/points', 10)

        # State
        self.prev_time = None
        self.prev_x = None
        self.prev_y = None
        self.prev_q = None

        self.timer = self.create_timer(1.0 / rate, self.timer_callback)

    def load_lidar(self, filepath):
        pc = np.fromfile(filepath, dtype=np.float32).reshape(-1, 5)
        return pc[:, :3]  # x, y, z only

    def create_pointcloud2(self, points, stamp):
        msg = PointCloud2()

        msg.header = Header()
        msg.header.stamp = stamp
        msg.header.frame_id = "lidar"

        msg.height = 1
        msg.width = points.shape[0]

        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = True

        msg.data = struct.pack('%sf' % (points.size), *points.flatten())

        return msg

    def timer_callback(self):
        if not self.sample_token:
            self.get_logger().info("Finished dataset")
            self.timer.cancel()
            return

        sample = self.nusc.get('sample', self.sample_token)
        lidar_token = sample['data']['LIDAR_TOP']
        sd = self.nusc.get('sample_data', lidar_token)
        ego_pose = self.nusc.get('ego_pose', sd['ego_pose_token'])

        # Load LiDAR
        lidar_path = os.path.join(self.nusc.dataroot, sd['filename'])
        points = self.load_lidar(lidar_path)

        ros_stamp = self.get_clock().now().to_msg()

        # Publish PointCloud
        pc_msg = self.create_pointcloud2(points, ros_stamp)
        self.pc_pub.publish(pc_msg)

        # Pose
        x, y = ego_pose['translation'][:2]
        qw, qx, qy, qz = ego_pose['rotation']

        dt = 0.0
        if self.prev_time:
            dt = (sd['timestamp'] - self.prev_time) / 1e6

        # Velocity (stable version)
        v = 0.0
        if dt > 0 and self.prev_x is not None:
            dx = x - self.prev_x
            dy = y - self.prev_y
            yaw = quat_to_yaw(qw, qx, qy, qz)
            v = (dx * math.cos(yaw) + dy * math.sin(yaw)) / dt

        # Yaw rate
        yaw_rate = 0.0
        if dt > 0 and self.prev_q is not None:
            yaw = quat_to_yaw(qw, qx, qy, qz)
            prev_yaw = quat_to_yaw(*self.prev_q)
            dyaw = math.atan2(math.sin(yaw - prev_yaw), math.cos(yaw - prev_yaw))
            yaw_rate = dyaw / dt

        # IMU
        imu = Imu()
        imu.header.stamp = ros_stamp
        imu.header.frame_id = "imu_link"
        imu.angular_velocity.z = yaw_rate
        imu.angular_velocity_covariance[8] = 0.01
        imu.orientation_covariance[0] = -1.0
        self.imu_pub.publish(imu)

        # Pose
        pose = PoseStamped()
        pose.header.stamp = ros_stamp
        pose.header.frame_id = "map"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = qw
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        self.pose_pub.publish(pose)

        # Odom
        odom = Odometry()
        odom.header.stamp = ros_stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose = pose.pose
        odom.twist.twist.linear.x = v
        self.odom_pub.publish(odom)

        # Update
        self.prev_time = sd['timestamp']
        self.prev_x = x
        self.prev_y = y
        self.prev_q = (qw, qx, qy, qz)

        self.sample_token = sample['next']


def main(args=None):
    rclpy.init(args=args)
    node = NuScenesBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()