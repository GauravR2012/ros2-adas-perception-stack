import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Time


class FakeOdomPublisher(Node):
    """
    Publishes a fake /odom stream for EKF testing.

    Improvements over the original:
      - Proper header: stamp + frame_id / child_frame_id (required by EKF and RViz)
      - Twist covariance populated so the EKF knows how noisy this source is
      - Configurable velocity and publish rate via ROS parameters
      - Optional sinusoidal velocity profile to exercise the filter under
        changing inputs (enable with use_sine_profile:=true)
    """

    def __init__(self):
        super().__init__('fake_odom_pub')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('velocity',         1.5)    # base linear velocity [m/s]
        self.declare_parameter('publish_rate_hz',  50.0)   # Hz  (match real odometry)
        self.declare_parameter('use_sine_profile', False)  # vary v sinusoidally

        v         = self.get_parameter('velocity').get_parameter_value().double_value
        rate_hz   = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        self._use_sine = self.get_parameter('use_sine_profile').get_parameter_value().bool_value

        self._base_v = v
        self._t      = 0.0
        self._dt     = 1.0 / rate_hz

        # ── Publisher ─────────────────────────────────────────────────────────
        self._pub = self.create_publisher(Odometry, '/odom', 10)
        self._timer = self.create_timer(self._dt, self._publish)

        self.get_logger().info(
            f"FakeOdomPublisher started — v={v:.2f} m/s  rate={rate_hz:.0f} Hz  "
            f"sine_profile={self._use_sine}"
        )

    # ── Timer callback ────────────────────────────────────────────────────────
    def _publish(self):
        # Optionally vary velocity to stress-test the EKF velocity channel
        if self._use_sine:
            v = self._base_v + 0.5 * math.sin(2.0 * math.pi * 0.2 * self._t)
        else:
            v = self._base_v
        self._t += self._dt

        msg = Odometry()

        # FIX: always stamp messages — EKF uses this for dt computation.
        # Without a stamp the EKF receives time=0, producing a bad dt on the
        # first tick and potentially a NaN/Inf in the prediction step.
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'        # fixed world frame
        msg.child_frame_id  = 'base_link'   # moving robot frame

        msg.twist.twist.linear.x = v

        # FIX: populate twist covariance (6×6 row-major).
        # The EKF currently uses a hard-coded r_vel param, but populating this
        # lets future versions read noise directly from the message, and it is
        # required by some Nav2 / SLAM consumers.
        #   index 0  → vx variance
        #   index 35 → ωz variance
        vel_variance   = 0.05 ** 2   # σ = 0.05 m/s  → σ² = 0.0025 m²/s²
        omega_variance = 0.01 ** 2   # σ = 0.01 rad/s
        msg.twist.covariance[0]  = vel_variance
        msg.twist.covariance[35] = omega_variance

        self._pub.publish(msg)


# ── Entry point ───────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = FakeOdomPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()