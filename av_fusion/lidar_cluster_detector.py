import rclpy

from rclpy.node import Node



from sensor_msgs.msg import PointCloud2

from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D



import numpy as np

from sensor_msgs_py import point_cloud2

from sklearn.cluster import DBSCAN





class LidarClusterDetector(Node):



    def __init__(self):

        super().__init__("lidar_cluster_detector")



        self.sub = self.create_subscription(

            PointCloud2,

            "/lidar/points",

            self.callback,

            10

        )



        self.pub = self.create_publisher(

            Detection3DArray,

            "/detections/boxes_3d",

            10

        )



        self.get_logger().info("🚀 LiDAR Clustering Detector (No TF, Stable)")



    # ---------------------------------------------------

    def callback(self, msg):



        # -------- Convert PointCloud2 → numpy --------

        points = []



        for p in point_cloud2.read_points(msg, skip_nans=True):

            points.append([p[0], p[1], p[2]])



        if len(points) == 0:

            return



        points = np.array(points)



        # -------- RANGE FILTER --------

        dist = np.linalg.norm(points[:, :2], axis=1)



        mask = (

            (dist < 40) &        # max range

            (points[:, 0] > 0) & # only front

            (np.abs(points[:, 1]) < 15)  # lane width approx

        )



        points = points[mask]



        if len(points) < 50:

            return



        # -------- GROUND REMOVAL --------

        points = points[points[:, 2] > -1.5]



        # -------- DOWNSAMPLE --------

        points = points[::3]



        # -------- CLUSTERING --------

        clustering = DBSCAN(eps=0.8, min_samples=15).fit(points)

        labels = clustering.labels_



        detections = Detection3DArray()

        detections.header = msg.header



        unique_labels = set(labels)



        for lbl in unique_labels:



            if lbl == -1:

                continue



            cluster = points[labels == lbl]



            if len(cluster) < 20:

                continue



            # -------- BOUNDING BOX --------

            min_bound = cluster.min(axis=0)

            max_bound = cluster.max(axis=0)



            center = (min_bound + max_bound) / 2

            size = max_bound - min_bound



            # -------- FILTER BAD CLUSTERS --------

            if size[0] < 0.5 or size[1] < 0.5:

                continue



            if size[0] > 6 or size[1] > 6:

                continue



            det = Detection3D()

            det.header = msg.header



            bbox = BoundingBox3D()



            bbox.center.position.x = float(center[0])

            bbox.center.position.y = float(center[1])

            bbox.center.position.z = float(center[2])



            bbox.size.x = float(size[0])

            bbox.size.y = float(size[1])

            bbox.size.z = float(size[2])



            det.bbox = bbox



            detections.detections.append(det)



        self.pub.publish(detections)





def main():

    rclpy.init()

    node = LidarClusterDetector()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()





if __name__ == "__main__":

    main()

