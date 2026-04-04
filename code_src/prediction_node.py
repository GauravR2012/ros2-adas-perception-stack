
# import rclpy
# from rclpy.node import Node

# from visualization_msgs.msg import Marker, MarkerArray
# from geometry_msgs.msg import Point


# class PredictionNode(Node):

#     def __init__(self):
#         super().__init__("prediction_node")

#         self.sub = self.create_subscription(
#             MarkerArray,
#             "/lidar/detection_markers",
#             self.callback,
#             10
#         )

#         self.pub = self.create_publisher(
#             MarkerArray,
#             "/predictions",
#             10
#         )

#         self.get_logger().info("🚀 Prediction Node Started")

#     def callback(self, msg):

#         pred_array = MarkerArray()

#         for marker in msg.markers:

#             if marker.ns != "tracked":
#                 continue

#             vx = 1.0  # placeholder (since velocity not in marker)
#             vy = 0.0

#             x = marker.pose.position.x
#             y = marker.pose.position.y

#             for i in range(5):
#                 px = x + vx * (i * 0.5)
#                 py = y + vy * (i * 0.5)

#                 m = Marker()
#                 m.header = marker.header
#                 m.ns = "prediction"
#                 m.id = marker.id * 100 + i
#                 m.type = Marker.SPHERE
#                 m.action = Marker.ADD

#                 m.pose.position.x = px
#                 m.pose.position.y = py
#                 m.pose.position.z = 0.5

#                 m.scale.x = 0.3
#                 m.scale.y = 0.3
#                 m.scale.z = 0.3

#                 m.color.r = 1.0
#                 m.color.a = 0.6

#                 pred_array.markers.append(m)

#         self.pub.publish(pred_array)


# def main():
#     rclpy.init()
#     node = PredictionNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == "__main__":
#     main()

import rclpy

from rclpy.node import Node



from vision_msgs.msg import Detection3DArray

from visualization_msgs.msg import Marker, MarkerArray





class PredictionNode(Node):



    def __init__(self):

        super().__init__("prediction_node")



        self.sub = self.create_subscription(

            Detection3DArray,

            "/tracked_objects",

            self.callback,

            10

        )



        self.pub = self.create_publisher(

            MarkerArray,

            "/predictions",

            10

        )



        self.get_logger().info("🚀 Prediction Node (REAL) Started")



    def callback(self, msg):



        marker_array = MarkerArray()



        for i, det in enumerate(msg.detections):



            x = det.bbox.center.position.x

            y = det.bbox.center.position.y



            vx = det.bbox.size.x

            vy = det.bbox.size.y



            for step in range(10):

                t = step * 0.5



                px = x + vx * t

                py = y + vy * t



                m = Marker()

                m.header = msg.header

                m.ns = "prediction"

                m.id = i * 100 + step

                m.type = Marker.SPHERE

                m.action = Marker.ADD



                m.pose.position.x = px

                m.pose.position.y = py

                m.pose.position.z = 0.5



                m.scale.x = 0.3

                m.scale.y = 0.3

                m.scale.z = 0.3



                m.color.r = 1.0

                m.color.a = 0.6



                marker_array.markers.append(m)



        self.pub.publish(marker_array)





def main():

    rclpy.init()

    node = PredictionNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()





if __name__ == "__main__":

    main()


