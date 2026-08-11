from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get package directories
    ekf_pkg = get_package_share_directory('ekf_localization_cpp')
    ekf_params = os.path.join(ekf_pkg, 'config', 'ekf_params.yaml')

    # Declare launch configuration variables
    dataset_path_arg = DeclareLaunchArgument(
        'dataset_path',
        default_value='/home/adarsh/av_perception/data/nuscenes',
        description='Path to the nuScenes mini dataset root'
    )
    
    dataset_path = LaunchConfiguration('dataset_path')

    return LaunchDescription([
        # 1. Dataset path argument
        dataset_path_arg,

        # 2. NuScenes Bridge (Player)
        Node(
            package='av_fusion',
            executable='nuscenes_bridge',
            name='nuscenes_bridge',
            parameters=[{
                'dataset_path': dataset_path,
                'version': 'v1.0-mini',
                'playback_rate_hz': 10.0,
                'scene_index': 0
            }],
            output='screen'
        ),

        # 3. EKF Localization Node (Pose-driven mode)
        Node(
            package='ekf_localization_cpp',
            executable='ekf_node',
            name='ekf_localization_cpp',
            parameters=[
                ekf_params,
                {
                    'pose_driven_mode': True,
                    'predict_rate_hz': 10.0
                }
            ],
            output='screen'
        ),

        # 4. PointPillars LiDAR Detector (3D deep learning object detection)
        Node(
            package='av_fusion',
            executable='pointpillars_detector',
            name='pointpillars_detector',
            remappings=[('/lidar/points', '/points')],
            output='screen'
        ),

        # 5. Centralized Obstacle Tracker
        Node(
            package='av_fusion',
            executable='gt_tracker',
            name='gt_tracker',
            output='screen'
        ),

        # 6. Trajectory Prediction Node
        Node(
            package='av_fusion',
            executable='prediction_node',
            name='prediction_node',
            output='screen'
        ),

        # 7. Collision Estimator Node
        Node(
            package='av_fusion',
            executable='collision_estimator',
            name='collision_estimator',
            output='screen'
        ),

        # 8. ADAS Decision Node
        Node(
            package='av_fusion',
            executable='decision_node',
            name='decision_node',
            output='screen'
        ),

        # 9. RViz2 Visualisation
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', '/home/adarsh/ros2_ws/track.rviz'],
            output='screen'
        )
    ])
