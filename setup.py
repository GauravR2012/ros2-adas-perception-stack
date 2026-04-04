from setuptools import find_packages, setup

package_name = 'av_fusion'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='adarsh',
    maintainer_email='sharmaadarsh859@gmail.com',
    description='ADAS perception and tracking stack using ROS2 and nuScenes',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'nuscenes_player = av_fusion.nuscenes_player:main',
            'lidar_cluster_detector = av_fusion.lidar_cluster_detector:main',
            'lidar_detection_visualizer = av_fusion.lidar_detection_visualizer:main',
            'gt_tracker = av_fusion.gt_tracker_node:main',
            'gt_tracker_node = av_fusion.gt_tracker_node:main',
            'pointpillars_detector = av_fusion.pointpillars_detector_node:main',
            'centerpoint_detector = av_fusion.centerpoint_detector_node:main',
            'tracking_evaluator = av_fusion.tracking_evaluator_node:main',
            'fusion_node = av_fusion.fusion_node:main',
            'prediction_node = av_fusion.prediction_node:main',
            'decision_node = av_fusion.decision_node:main',
            'camera_detector = av_fusion.camera_detector_node:main',
            'kitti_player = av_fusion.kitti_player:main',
            'collision_estimator = av_fusion.collision_estimator:main',
        ],
    },
)