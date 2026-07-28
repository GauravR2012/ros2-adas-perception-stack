#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np

import tf2_ros

class MultiCamBEVProjectionNode(Node):
    """
    Surround 360-degree Bird's-Eye-View (BEV) Projection, Feature Extractor & Global Map Mosaic Node.
    - Publishes Local Ego-Centric BEV View (/bev/image_raw).
    - Publishes Extracted Road Features (/bev/features).
    - Accumulates & Stitches a Persistent Global BEV Map (/bev/global_map) as the vehicle drives!
    """
    def __init__(self):
        super().__init__('bev_projection_node')
        self.get_logger().info('Initializing Multi-Cam BEV & Global Map Mosaic Node...')

        self.bridge = CvBridge()

        # Canvas parameters
        self.declare_parameter('bev_output_topic', '/bev/image_raw')
        self.declare_parameter('feature_output_topic', '/bev/features')
        self.declare_parameter('global_map_topic', '/bev/global_map')
        self.declare_parameter('range_x_m', 30.0)      # Local longitudinal range
        self.declare_parameter('range_y_m', 30.0)      # Local lateral range
        self.declare_parameter('grid_resolution', 0.1) # 0.1m/pixel -> 300x300 local canvas
        self.declare_parameter('roi_radius_m', 14.0)   # Ground ROI cutoff

        self.bev_output_topic = self.get_parameter('bev_output_topic').value
        self.feature_output_topic = self.get_parameter('feature_output_topic').value
        self.global_map_topic = self.get_parameter('global_map_topic').value
        self.range_x = self.get_parameter('range_x_m').value
        self.range_y = self.get_parameter('range_y_m').value
        self.res = self.get_parameter('grid_resolution').value
        self.roi_radius = self.get_parameter('roi_radius_m').value

        # Local Canvas Dimensions (300 x 300 pixels)
        self.cols = int(self.range_y / self.res)
        self.rows = int(self.range_x / self.res)
        self.center_col = int(self.cols / 2)
        self.center_row = int(self.rows / 2)

        # Global Persistent BEV Mosaic Canvas (200m x 200m world area = 2000 x 2000 px)
        self.global_size_m = 200.0
        self.global_px = int(self.global_size_m / self.res)
        self.global_canvas = np.zeros((self.global_px, self.global_px, 3), dtype=np.uint8)
        self.global_center_px = self.global_px // 2
        self.initial_pose = None  # First vehicle pose reference
        self.latest_odom = None

        # Build Local Circular ROI Mask
        self.roi_mask = np.zeros((self.rows, self.cols), dtype=np.uint8)
        radius_px = int(self.roi_radius / self.res)
        cv2.circle(self.roi_mask, (self.center_col, self.center_row), radius_px, 255, -1)

        # TF Listener for Vehicle Pose
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribe to /odom as pose fallback
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # 6 Cameras Configuration
        self.cameras = {
            'front':       {'topic': '/camera/front/image_raw',       'yaw_deg': 0.0,    'x_off': 1.5,  'y_off': 0.0},
            'front_left':  {'topic': '/camera/front_left/image_raw',  'yaw_deg': 55.0,   'x_off': 1.2,  'y_off': 0.6},
            'front_right': {'topic': '/camera/front_right/image_raw', 'yaw_deg': -55.0,  'x_off': 1.2,  'y_off': -0.6},
            'back':        {'topic': '/camera/back/image_raw',        'yaw_deg': 180.0,  'x_off': -1.0, 'y_off': 0.0},
            'back_left':   {'topic': '/camera/back_left/image_raw',   'yaw_deg': 110.0,  'x_off': -0.5, 'y_off': 0.6},
            'back_right':  {'topic': '/camera/back_right/image_raw',  'yaw_deg': -110.0, 'x_off': -0.5, 'y_off': -0.6},
        }

        self.latest_images = {}
        self.H_matrices = {}

        # Publishers
        self.bev_pub = self.create_publisher(Image, self.bev_output_topic, 10)
        self.feature_pub = self.create_publisher(Image, self.feature_output_topic, 10)
        self.global_pub = self.create_publisher(Image, self.global_map_topic, 10)

        for name, cfg in self.cameras.items():
            self.create_subscription(
                Image, cfg['topic'],
                lambda msg, c_name=name: self.image_callback(msg, c_name),
                10
            )

        # 10 Hz Timer for composite & global map rendering
        self.timer = self.create_timer(0.1, self.generate_bev_composite)
        self.get_logger().info('🚀 Multi-Cam Local & Global BEV Mosaic Node initialized!')

    def odom_callback(self, msg):
        self.latest_odom = msg

    def compute_homography(self, cam_name, img_h, img_w):
        cfg = self.cameras[cam_name]
        yaw = np.radians(cfg['yaw_deg'])

        f_x = img_w / (2.0 * np.tan(np.radians(35.0)))
        f_y = f_x
        c_x = img_w / 2.0
        c_y = img_h * 0.55

        pts_cam_ground = np.array([
            [3.0,  -4.0, -1.5],
            [3.0,   4.0, -1.5],
            [14.0,  7.0, -1.5],
            [14.0, -7.0, -1.5]
        ])

        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        R_yaw = np.array([
            [cos_y, -sin_y, 0],
            [sin_y,  cos_y, 0],
            [0,      0,     1]
        ])

        src_pts = []
        dst_pts = []

        for pt in pts_cam_ground:
            u_img = int(f_x * (pt[1] / pt[0]) + c_x)
            v_img = int(f_y * (-pt[2] / pt[0]) + c_y)
            src_pts.append([u_img, v_img])

            pt_veh = R_yaw @ pt + np.array([cfg['x_off'], cfg['y_off'], 0.0])
            x_fwd = pt_veh[0]
            y_lat = pt_veh[1]

            col_bev = int((self.range_y / 2.0 - y_lat) / self.res)
            row_bev = int((self.range_x / 2.0 - x_fwd) / self.res)
            dst_pts.append([col_bev, row_bev])

        src_pts = np.float32(src_pts)
        dst_pts = np.float32(dst_pts)

        self.H_matrices[cam_name] = cv2.getPerspectiveTransform(src_pts, dst_pts)

    def image_callback(self, msg, cam_name):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_images[cam_name] = (cv_img, msg.header)
        except Exception as e:
            self.get_logger().error(f'Image decode error: {e}')

    def extract_road_features(self, bev_rgb):
        gray = cv2.cvtColor(bev_rgb, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, -5
        )
        feature_grid = cv2.bitwise_and(binary, self.roi_mask)
        cv2.rectangle(feature_grid, (self.center_col - 10, self.center_row - 18), (self.center_col + 10, self.center_row + 18), 0, -1)
        return feature_grid

    def get_vehicle_pose(self):
        """
        Attempts to get vehicle pose from TF (map/odom -> base_link), fallback to /odom.
        """
        for target_frame in ['map', 'odom']:
            try:
                t = self.tf_buffer.lookup_transform(target_frame, 'base_link', rclpy.time.Time())
                x = t.transform.translation.x
                y = t.transform.translation.y
                qx = t.transform.rotation.x
                qy = t.transform.rotation.y
                qz = t.transform.rotation.z
                qw = t.transform.rotation.w
                siny_cosp = 2.0 * (qw * qz + qx * qy)
                cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
                yaw = np.arctan2(siny_cosp, cosy_cosp)
                return x, y, yaw
            except Exception:
                continue

        if self.latest_odom is not None:
            p = self.latest_odom.pose.pose.position
            q = self.latest_odom.pose.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            return p.x, p.y, yaw

        return None

    def update_global_mosaic(self, local_bev_unicon):
        """
        Stitches current local BEV image onto the persistent global world mosaic.
        """
        pose = self.get_vehicle_pose()
        if pose is None:
            return

        x, y, yaw = pose

        if self.initial_pose is None:
            self.initial_pose = (x, y)

        rel_x = x - self.initial_pose[0]
        rel_y = y - self.initial_pose[1]

        center_g_col = int(self.global_center_px + rel_y / self.res)
        center_g_row = int(self.global_center_px - rel_x / self.res)

        yaw_deg = np.degrees(yaw)
        M_rot = cv2.getRotationMatrix2D((self.center_col, self.center_row), -yaw_deg, 1.0)
        rotated_bev = cv2.warpAffine(local_bev_unicon, M_rot, (self.cols, self.rows))

        r_min = center_g_row - self.center_row
        r_max = center_g_row + (self.rows - self.center_row)
        c_min = center_g_col - self.center_col
        c_max = center_g_col + (self.cols - self.center_col)

        if r_min >= 0 and r_max < self.global_px and c_min >= 0 and c_max < self.global_px:
            mask = rotated_bev > 0
            patch = self.global_canvas[r_min:r_max, c_min:c_max]
            patch[mask] = np.where(patch[mask] == 0, rotated_bev[mask], (patch[mask] // 2 + rotated_bev[mask] // 2))
            self.global_canvas[r_min:r_max, c_min:c_max] = patch

    def generate_bev_composite(self):
        if not self.latest_images:
            return

        bev_canvas = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        latest_header = None

        for cam_name, (img, header) in self.latest_images.items():
            latest_header = header
            h, w = img.shape[:2]

            if cam_name not in self.H_matrices:
                self.compute_homography(cam_name, h, w)

            H = self.H_matrices[cam_name]
            warped = cv2.warpPerspective(img, H, (self.cols, self.rows))

            mask = warped > 0
            bev_canvas[mask] = np.where(bev_canvas[mask] == 0, warped[mask], (bev_canvas[mask] // 2 + warped[mask] // 2))

        # Apply Ground ROI Mask
        bev_canvas = cv2.bitwise_and(bev_canvas, bev_canvas, mask=self.roi_mask)

        # Update persistent global mosaic map before drawing vehicle icon
        self.update_global_mosaic(bev_canvas)

        # Extract Binary Road Features
        feature_grid = self.extract_road_features(bev_canvas)

        # Draw vehicle icon in local center
        cv2.rectangle(bev_canvas, (self.center_col - 8, self.center_row - 15), (self.center_col + 8, self.center_row + 15), (0, 255, 0), -1)
        cv2.circle(bev_canvas, (self.center_col, self.center_row - 15), 3, (0, 0, 255), -1)

        # 1. Publish Local Ego-Centric BEV Image
        bev_msg = self.bridge.cv2_to_imgmsg(bev_canvas, encoding='bgr8')
        bev_msg.header = latest_header if latest_header else Header()
        bev_msg.header.frame_id = 'base_link'
        self.bev_pub.publish(bev_msg)

        # 2. Publish Binary Feature Grid
        feat_msg = self.bridge.cv2_to_imgmsg(feature_grid, encoding='mono8')
        feat_msg.header = bev_msg.header
        self.feature_pub.publish(feat_msg)

        # 3. Publish Persistent Global Stitched BEV Map
        global_msg = self.bridge.cv2_to_imgmsg(self.global_canvas, encoding='bgr8')
        global_msg.header = bev_msg.header
        global_msg.header.frame_id = 'map'
        self.global_pub.publish(global_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MultiCamBEVProjectionNode()
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
