import rclpy

from rclpy.node import Node



from vision_msgs.msg import Detection3DArray

from visualization_msgs.msg import Marker, MarkerArray



import numpy as np

import time





class Track:

    def __init__(self, id, position):

        self.id = id

        self.position = np.array(position)

        self.velocity = np.zeros(3)

        self.last_update = time.time()

        self.missed = 0





class StableTracker(Node):



    def __init__(self):

        super().__init__('stable_tracker')



        self.sub = self.create_subscription(

            Detection3DArray,

            '/detections/lidar_clusters',

            self.callback,

            10

        )



        self.pub = self.create_publisher(

            MarkerArray,

            '/tracking/markers',

            10

        )



        self.tracks = []

        self.next_id = 0



        # PARAMETERS

        self.max_distance = 2.5   # matching threshold

        self.max_missed = 5      # track deletion

        self.alpha = 0.6         # velocity smoothing



        self.get_logger().info("🚀 Stable Tracker Started")





    def callback(self, msg):



        detections = []



        for det in msg.detections:

            p = det.bbox.center.position

            detections.append(np.array([p.x, p.y, p.z]))



        detections = np.array(detections)



        if len(detections) == 0:

            return



        self.update_tracks(detections)

        self.publish_markers(msg.header)





    def update_tracks(self, detections):



        if len(self.tracks) == 0:

            for det in detections:

                self.tracks.append(Track(self.next_id, det))

                self.next_id += 1

            return



        assigned = set()



        # =========================

        # DATA ASSOCIATION (GREEDY)

        # =========================

        for track in self.tracks:



            min_dist = float('inf')

            best_idx = -1



            for i, det in enumerate(detections):

                if i in assigned:

                    continue



                dist = np.linalg.norm(det - track.position)



                if dist < min_dist:

                    min_dist = dist

                    best_idx = i



            # =========================

            # MATCH FOUND

            # =========================

            if min_dist < self.max_distance:



                det = detections[best_idx]



                dt = time.time() - track.last_update

                if dt > 0:



                    measured_vel = (det - track.position) / dt



                    # 🔥 SMOOTHED VELOCITY

                    track.velocity = (

                        self.alpha * measured_vel +

                        (1 - self.alpha) * track.velocity

                    )



                track.position = det

                track.last_update = time.time()

                track.missed = 0



                assigned.add(best_idx)



            else:

                track.missed += 1



        # =========================

        # CREATE NEW TRACKS

        # =========================

        for i, det in enumerate(detections):

            if i not in assigned:

                self.tracks.append(Track(self.next_id, det))

                self.next_id += 1



        # =========================

        # DELETE LOST TRACKS

        # =========================

        self.tracks = [t for t in self.tracks if t.missed < self.max_missed]





    def publish_markers(self, header):



        marker_array = MarkerArray()



        for track in self.tracks:



            # =========================

            # BOX MARKER

            # =========================

            marker = Marker()

            marker.header = header

            marker.ns = "tracks"

            marker.id = track.id

            marker.type = Marker.CUBE

            marker.action = Marker.ADD



            marker.pose.position.x = float(track.position[0])

            marker.pose.position.y = float(track.position[1])

            marker.pose.position.z = float(track.position[2])



            marker.scale.x = 2.0

            marker.scale.y = 1.0

            marker.scale.z = 1.5



            marker.color.r = 1.0

            marker.color.g = 0.0

            marker.color.b = 0.0

            marker.color.a = 0.8



            marker_array.markers.append(marker)



            # =========================

            # VELOCITY ARROW

            # =========================

            arrow = Marker()

            arrow.header = header

            arrow.ns = "velocity"

            arrow.id = track.id + 1000

            arrow.type = Marker.ARROW

            arrow.action = Marker.ADD



            arrow.pose.position.x = float(track.position[0])

            arrow.pose.position.y = float(track.position[1])

            arrow.pose.position.z = float(track.position[2])



            arrow.scale.x = np.linalg.norm(track.velocity) * 2.0

            arrow.scale.y = 0.3

            arrow.scale.z = 0.3



            arrow.color.r = 0.0

            arrow.color.g = 0.0

            arrow.color.b = 1.0

            arrow.color.a = 1.0



            marker_array.markers.append(arrow)



        self.pub.publish(marker_array)





def main(args=None):

    rclpy.init(args=args)

    node = StableTracker()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()