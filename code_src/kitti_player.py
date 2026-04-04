import rclpy

from rclpy.node import Node



from sensor_msgs.msg import Image, PointCloud2

from visualization_msgs.msg import MarkerArray, Marker

from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D

from geometry_msgs.msg import TransformStamped



from cv_bridge import CvBridge

import tf2_ros



import numpy as np

import cv2

import os



from sensor_msgs_py import point_cloud2





class KittiPlayer(Node):



    def __init__(self):

        super().__init__("kitti_player")



        # 🔥 CHANGE THIS PATH

        self.kitti_root = "/home/adarsh/kitti"



        self.velodyne_path = os.path.join(self.kitti_root, "velodyne")

        self.image_path = os.path.join(self.kitti_root, "image_2")

        self.label_path = os.path.join(self.kitti_root, "label_2")



        self.files = sorted(os.listdir(self.velodyne_path))

        self.index = 0



        self.bridge = CvBridge()



        # ---------------- Publishers ----------------

        self.lidar_pub = self.create_publisher(PointCloud2, "/lidar/points", 10)

        self.image_pub = self.create_publisher(Image, "/camera/front/image", 10)

        self.gt_pub = self.create_publisher(Detection3DArray, "/detections/gt_boxes", 10)

        self.marker_pub = self.create_publisher(MarkerArray, "/gt/markers", 10)



        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)



        self.timer = self.create_timer(0.5, self.timer_callback)



        self.get_logger().info("🚀 KITTI Player Started")



    # ---------------------------------------------------

    def publish_tf(self, timestamp):



        t = TransformStamped()

        t.header.stamp = timestamp

        t.header.frame_id = "map"

        t.child_frame_id = "base_link"



        # KITTI has no ego motion → static

        t.transform.translation.x = 0.0

        t.transform.translation.y = 0.0

        t.transform.translation.z = 0.0



        t.transform.rotation.w = 1.0



        self.tf_broadcaster.sendTransform(t)



    # ---------------------------------------------------

    def publish_lidar(self, file, timestamp):



        points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)



        xyz = points[:, :3]



        header = self.get_clock().now().to_msg()

        cloud = point_cloud2.create_cloud_xyz32(

            header,

            xyz

        )



        cloud.header.frame_id = "base_link"

        self.lidar_pub.publish(cloud)



    # ---------------------------------------------------

    def publish_image(self, file, timestamp):



        img = cv2.imread(file)



        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")

        msg.header.stamp = timestamp

        msg.header.frame_id = "camera_front"



        self.image_pub.publish(msg)



    # ---------------------------------------------------

    def publish_gt(self, label_file, timestamp):



        detections = Detection3DArray()

        detections.header.stamp = timestamp

        detections.header.frame_id = "base_link"



        marker_array = MarkerArray()



        if not os.path.exists(label_file):

            return



        with open(label_file, "r") as f:

            lines = f.readlines()



        for i, line in enumerate(lines):



            data = line.strip().split()



            obj_type = data[0]



            if obj_type not in ["Car", "Pedestrian", "Cyclist"]:

                continue



            h = float(data[8])

            w = float(data[9])

            l = float(data[10])

            x = float(data[11])

            y = float(data[12])

            z = float(data[13])



            det = Detection3D()

            det.header = detections.header



            bbox = BoundingBox3D()

            bbox.center.position.x = x

            bbox.center.position.y = y

            bbox.center.position.z = z



            bbox.size.x = l

            bbox.size.y = w

            bbox.size.z = h



            det.bbox = bbox

            detections.detections.append(det)



            # -------- Visualization --------

            m = Marker()

            m.header = detections.header

            m.ns = "gt"

            m.id = i

            m.type = Marker.CUBE

            m.action = Marker.ADD



            m.pose.position.x = x

            m.pose.position.y = y

            m.pose.position.z = z



            m.scale.x = l

            m.scale.y = w

            m.scale.z = h



            m.color.g = 1.0

            m.color.a = 0.5



            marker_array.markers.append(m)



        self.gt_pub.publish(detections)

        self.marker_pub.publish(marker_array)



    # ---------------------------------------------------

    def timer_callback(self):



        if self.index >= len(self.files):

            self.index = 0



        file = self.files[self.index].split(".")[0]



        lidar_file = os.path.join(self.velodyne_path, file + ".bin")

        image_file = os.path.join(self.image_path, file + ".png")

        label_file = os.path.join(self.label_path, file + ".txt")



        timestamp = self.get_clock().now().to_msg()



        self.publish_tf(timestamp)

        self.publish_lidar(lidar_file, timestamp)

        self.publish_image(image_file, timestamp)

        self.publish_gt(label_file, timestamp)



        self.index += 1





def main():

    rclpy.init()

    node = KittiPlayer()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()





if __name__ == "__main__":

    main()

