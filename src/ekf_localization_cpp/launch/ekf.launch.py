from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg = get_package_share_directory('ekf_localization_cpp')
    ekf_params  = os.path.join(pkg, 'config', 'ekf_params.yaml')
    slam_params = os.path.join(pkg, 'config', 'slam_params.yaml')

    return LaunchDescription([
        # 1. Mock IMU + odometry sensors
        Node(package='ekf_localization_cpp',
             executable='mock_sensors',
             name='mock_sensors',
             output='screen'),

        # 2. Fake LiDAR for SLAM input
        Node(package='ekf_localization_cpp',
             executable='fake_laser',
             name='fake_laser',
             output='screen'),

        # 3. SLAM — builds map, publishes map→odom TF
        Node(package='slam_toolbox',
             executable='async_slam_toolbox_node',
             name='slam_toolbox',
             parameters=[slam_params],
             output='screen'),

        # 4. Bridge — converts map→base_link TF to /pose_measurement
        Node(package='ekf_localization_cpp',
             executable='slam_pose_bridge',
             name='slam_pose_bridge',
             output='screen'),

        # 5. EKF — fuses odom + IMU + SLAM pose
        Node(package='ekf_localization_cpp',
             executable='ekf_node',
             name='ekf_localization_cpp',
             parameters=[ekf_params],
             output='screen'),
    ])
