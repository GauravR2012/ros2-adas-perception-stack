from launch import LaunchDescription
from launch_ros.actions import Node, LifecycleNode
from launch.actions import TimerAction, EmitEvent
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
from ament_index_python.packages import get_package_share_directory
import launch_ros.events.lifecycle
import os


def generate_launch_description():
    pkg = get_package_share_directory('ekf_localization_cpp')
    ekf_params  = os.path.join(pkg, 'config', 'ekf_params.yaml')
    slam_params = os.path.join(pkg, 'config', 'slam_params.yaml')

    slam_node = LifecycleNode(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        namespace='',
        parameters=[slam_params],
        output='screen',
    )

    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=launch_ros.events.lifecycle.matches_node_name(
                node_name='/slam_toolbox'
            ),
            transition_id=Transition.TRANSITION_CONFIGURE,
        )
    )

    activate_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=launch_ros.events.lifecycle.matches_node_name(
                node_name='/slam_toolbox'
            ),
            transition_id=Transition.TRANSITION_ACTIVATE,
        )
    )

    return LaunchDescription([

        # 1. Static TF: base_link → lidar
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_lidar_top',
            arguments=[
                '--x', '0.0', '--y', '0.0', '--z', '1.84',
                '--roll', '0.0', '--pitch', '0.0', '--yaw', '0.0',
                '--frame-id', 'base_link',
                '--child-frame-id', 'lidar',
            ],
            output='screen',
        ),

        # 2. PointCloud2 → LaserScan
        Node(
            package='ekf_localization_cpp',
            executable='pointcloud_to_laserscan',
            name='pointcloud_to_laserscan',
            parameters=[{
                'min_z': -0.3, 'max_z': 1.5,
                'range_min': 0.3, 'range_max': 50.0,
                'angle_min': -3.14159, 'angle_max': 3.14159,
                'angle_increment': 0.00436332,
                'scan_frame': 'lidar',
                'input_topic': '/lidar/points',
                'output_topic': '/scan',
            }],
            output='screen',
        ),

        # 3. EKF node — publishes odom→base_link TF
        # NOTE: pose_driven_mode is read from ekf_params.yaml (true)
        # Do NOT add an inline override here — it would overwrite the yaml value
        Node(
            package='ekf_localization_cpp',
            executable='ekf_node',
            name='ekf_localization_cpp',
            parameters=[ekf_params],
            output='screen',
        ),

        # 4. SLAM Toolbox (lifecycle node — starts in unconfigured state)
        slam_node,

        # 5. After 4 s: configure SLAM (EKF should have odom→base_link by now)
        TimerAction(period=4.0, actions=[configure_event]),

        # 6. After 5 s: activate SLAM (starts subscribing to /scan, building map)
        TimerAction(period=5.0, actions=[activate_event]),

        # 7. After 6 s: start SLAM pose bridge (reads map→base_link TF)
        TimerAction(period=6.0, actions=[
            Node(
                package='ekf_localization_cpp',
                executable='slam_pose_bridge',
                name='slam_pose_bridge',
                output='screen',
            ),
        ]),
    ])