
import rclpy

from rclpy.node import Node



from vision_msgs.msg import Detection3DArray

from geometry_msgs.msg import Twist



import numpy as np

import time





class CollisionEstimator(Node):



    def __init__(self):

        super().__init__("collision_estimator")



        self.sub = self.create_subscription(

            Detection3DArray,

            "/detections/boxes_3d",

            self.callback,

            10

        )



        self.cmd_pub = self.create_publisher(

            Twist,

            "/cmd_vel",

            10

        )



        self.prev_objects = None

        self.prev_time = None



        self.get_logger().info("🚨 Collision Estimator (FINAL STABLE)")





    # ---------------------------------------------------

    def callback(self, msg):



        current_time = time.time()



        # -------- Extract detections --------

        objects = []

        for det in msg.detections:

            x = det.bbox.center.position.x

            y = det.bbox.center.position.y

            objects.append(np.array([x, y]))



        if len(objects) == 0:

            return



        objects = np.array(objects)



        # -------- First frame --------

        if self.prev_objects is None:

            self.prev_objects = objects

            self.prev_time = current_time

            return



        # -------- Compute dt --------

        dt = current_time - self.prev_time



        # 🚨 CRITICAL FIX

        if dt < 0.05:

            return



        velocities = []



        # ---------------------------------------------------

        # 🔹 NEAREST NEIGHBOR MATCHING (with gating)

        # ---------------------------------------------------

        for curr in objects:



            dists = np.linalg.norm(self.prev_objects - curr, axis=1)



            idx = np.argmin(dists)



            # 🚨 MATCHING FILTER

            if dists[idx] > 3.0:

                continue



            prev = self.prev_objects[idx]



            vel = (curr - prev) / dt



            # 🚨 VELOCITY CLAMP

            if np.linalg.norm(vel) > 20:

                continue



            velocities.append((curr, vel))



        # ---------------------------------------------------

        # 🔹 TTC COMPUTATION

        # ---------------------------------------------------

        brake = False



        for i, (obj, vel) in enumerate(velocities):



            vx = vel[0]

            distance = np.linalg.norm(obj)



            # 👉 Only front objects

            if obj[0] < 0:

                continue



            # 👉 Lane filtering (IMPORTANT)

            if abs(obj[1]) > 2.5:

                continue



            # 👉 Only approaching objects

            if vx >= -0.1:

                continue



            ttc = distance / abs(vx)



            self.get_logger().info(

                f"Obj {i} | Dist: {distance:.2f} | Vel: {vx:.2f} | TTC: {ttc:.2f}"

            )



            if ttc < 3.0:

                brake = True



        # ---------------------------------------------------

        # 🔹 CONTROL OUTPUT

        # ---------------------------------------------------

        cmd = Twist()



        if brake:

            cmd.linear.x = 0.0

            self.get_logger().warn("🚨 BRAKE APPLIED")

        else:

            cmd.linear.x = 2.0



        self.cmd_pub.publish(cmd)



        # -------- Update history --------

        self.prev_objects = objects

        self.prev_time = current_time





# ---------------------------------------------------

def main():

    rclpy.init()

    node = CollisionEstimator()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()





if __name__ == "__main__":

    main()