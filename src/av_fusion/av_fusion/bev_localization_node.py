#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np

class BEVLocalizationNode(Node):
    """
    BEV Feature Matching Localization Node.
    - Subscribes to extracted BEV road features (/bev/features).
    - Subscribes to global Occupancy Map (/map) and current estimated vehicle pose (/odom).
    - Uses 2D Template Matching / ICP registration to compute visual map alignment correction.
    - Publishes refined pose measurement (/bev_pose) for EKF fusion.
    """
    def __init__(self):
        super().__init__('bev_localization_node')
        self.get_logger().info('Initializing BEV Feature Matching Localization Engine...')

        self.bridge = CvBridge()

        # Configurable Parameters
        self.declare_parameter('feature_topic', '/bev/features')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('bev_pose_topic', '/bev_pose')
        self.declare_parameter('grid_resolution', 0.1) # 0.1m per pixel

        self.feature_topic = self.get_parameter('feature_topic').value
        self.map_topic = self.get_parameter('map_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.bev_pose_topic = self.get_parameter('bev_pose_topic').value
        self.res = self.get_parameter('grid_resolution').value

        # State Variables
        self.latest_feature_grid = None
        self.latest_odom_pose = None
        self.global_map = None
        self.map_info = None

        # Subscriptions & Publishers
        self.feat_sub = self.create_subscription(Image, self.feature_topic, self.feature_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_callback, 10)

        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, self.bev_pose_topic, 10)

        # 5 Hz Matching Loop
        self.timer = self.create_timer(0.2, self.perform_feature_matching)
        self.get_logger().info('🚀 BEV Feature Matching Localization active!')

    def feature_callback(self, msg):
        try:
            self.latest_feature_grid = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f'Error reading feature grid: {e}')

    def odom_callback(self, msg):
        self.latest_odom_pose = msg.pose.pose

    def map_callback(self, msg):
        self.map_info = msg.info
        map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        # Convert occupancy grid (0=free, 100=occupied) to binary image (255=obstacle/marking)
        self.global_map = np.where(map_data > 50, 255, 0).astype(np.uint8)

    def perform_feature_matching(self):
        if self.latest_feature_grid is None or self.latest_odom_pose is None or self.global_map is None:
            return

        # Vehicle position in Odom/Map frame
        x_est = self.latest_odom_pose.position.x
        y_est = self.latest_odom_pose.position.y

        # Convert vehicle pose to global map pixel coords
        map_origin_x = self.map_info.origin.position.x
        map_origin_y = self.map_info.origin.position.y
        map_res = self.map_info.resolution

        center_map_col = int((x_est - map_origin_x) / map_res)
        center_map_row = int((y_est - map_origin_y) / map_res)

        # Crop local sub-map patch around estimated pose (e.g., 60m x 60m)
        patch_size_px = int(60.0 / map_res)
        half_patch = patch_size_px // 2

        r_min = max(0, center_map_row - half_patch)
        r_max = min(self.global_map.shape[0], center_map_row + half_patch)
        c_min = max(0, center_map_col - half_patch)
        c_max = min(self.global_map.shape[1], center_map_col + half_patch)

        map_patch = self.global_map[r_min:r_max, c_min:c_max]

        if map_patch.shape[0] < self.latest_feature_grid.shape[0] or map_patch.shape[1] < self.latest_feature_grid.shape[1]:
            return

        # 2D Template Matching between BEV feature template and map patch
        res_match = cv2.matchTemplate(map_patch, self.latest_feature_grid, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_match)

        # High confidence match threshold
        if max_val > 0.35:
            # Match top-left corner
            match_c, match_r = max_loc
            
            # Corrected center position in patch
            corrected_c = match_c + self.latest_feature_grid.shape[1] // 2
            corrected_r = match_r + self.latest_feature_grid.shape[0] // 2

            # Offset translation from predicted center
            delta_col = corrected_c - half_patch
            delta_row = corrected_r - half_patch

            delta_x = delta_row * map_res
            delta_y = delta_col * map_res

            # Refined metric pose
            x_corrected = x_est + delta_x
            y_corrected = y_est + delta_y

            # Publish Refined BEV Pose for EKF Update
            bev_pose_msg = PoseWithCovarianceStamped()
            bev_pose_msg.header = Header()
            bev_pose_msg.header.stamp = self.get_clock().now().to_msg()
            bev_pose_msg.header.frame_id = 'map'

            bev_pose_msg.pose.pose.position.x = float(x_corrected)
            bev_pose_msg.pose.pose.position.y = float(y_corrected)
            bev_pose_msg.pose.pose.position.z = 0.0
            bev_pose_msg.pose.pose.orientation = self.latest_odom_pose.orientation

            # Covariance matrix (low covariance = high confidence match)
            bev_pose_msg.pose.covariance[0] = 0.05  # var(x)
            bev_pose_msg.pose.covariance[7] = 0.05  # var(y)
            bev_pose_msg.pose.covariance[35] = 0.02 # var(yaw)

            self.pose_pub.publish(bev_pose_msg)
            self.get_logger().info(f'🎯 BEV Feature Match score: {max_val:.2f} | Pose correction: Δx={delta_x:.2f}m, Δy={delta_y:.2f}m')

def main(args=None):
    rclpy.init(args=args)
    node = BEVLocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
