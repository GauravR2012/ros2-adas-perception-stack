#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <random>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// MockSensorPublisher
//
// Simulates a robot driving a circle of radius R at angular velocity w.
// Publishes three independent noisy sensor streams at realistic rates:
//   • /imu           — 100 Hz, low noise + small constant bias
//   • /odom          — 50 Hz,  medium noise on linear velocity
//   • /pose_measurement — 10 Hz, high noise (GPS / VIO quality)
//
// Use for EKF development and Q/R tuning without hardware.
// Ground truth is printed to /tf_static so you can compare in RViz.
// ─────────────────────────────────────────────────────────────────────────────
class MockSensorPublisher : public rclcpp::Node
{
public:
    MockSensorPublisher()
    : Node("mock_sensors"),
      t_(0.0),
      gen_(42),                          // fixed seed → reproducible runs
      imu_noise_(0.0,  0.01),            // σ = 0.01 rad/s
      odom_noise_(0.0, 0.05),            // σ = 0.05 m/s
      pose_noise_(0.0, 0.20)             // σ = 0.20 m  (GPS-grade)
    {
        // ── Declare parameters ────────────────────────────────────────────────
        declare_parameter("radius",    3.0);   // circle radius [m]
        declare_parameter("omega",     0.3);   // angular velocity [rad/s]
        declare_parameter("imu_bias",  0.005); // constant yaw-rate bias [rad/s]

        odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
        imu_pub_  = create_publisher<sensor_msgs::msg::Imu>("/imu",   10);
        pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/pose_measurement", 10);

        // IMU at 100 Hz — dt = 0.01 s
        imu_timer_ = create_wall_timer(std::chrono::milliseconds(10),
            [this]() {
                publishImu();
                t_ += 0.01;          // advance ground-truth clock on IMU tick
            });

        // Odometry at 50 Hz
        odom_timer_ = create_wall_timer(std::chrono::milliseconds(20),
            [this]() { publishOdom(); });

        // Pose at 10 Hz
        pose_timer_ = create_wall_timer(std::chrono::milliseconds(100),
            [this]() { publishPose(); });

        RCLCPP_INFO(get_logger(), "MockSensorPublisher started — circle r=%.1f m  ω=%.2f rad/s",
            get_parameter("radius").as_double(),
            get_parameter("omega").as_double());
    }

private:
    // ── Ground truth at time t ────────────────────────────────────────────────
    struct GroundTruth { double x, y, yaw, v, yaw_rate; };

    GroundTruth groundTruth(double t) const
    {
        const double R = get_parameter("radius").as_double();
        const double w = get_parameter("omega").as_double();
        GroundTruth gt;
        gt.x        = R * std::cos(w * t);
        gt.y        = R * std::sin(w * t);
        gt.yaw      = w * t + M_PI / 2.0;
        gt.v        = R * w;
        gt.yaw_rate = w;
        return gt;
    }

    // ── IMU ───────────────────────────────────────────────────────────────────
    void publishImu()
    {
        const double bias = get_parameter("imu_bias").as_double();
        auto gt = groundTruth(t_);

        sensor_msgs::msg::Imu msg;
        msg.header.stamp    = now();
        msg.header.frame_id = "imu_link";

        // Yaw rate: true value + Gaussian noise + constant bias
        msg.angular_velocity.z = gt.yaw_rate + imu_noise_(gen_) + bias;

        // Linear acceleration (centripetal, pointing inward on a circle)
        msg.linear_acceleration.x = gt.v * gt.yaw_rate + imu_noise_(gen_);
        msg.linear_acceleration.y = 0.0;
        msg.linear_acceleration.z = 9.81;  // gravity component

        // Covariance diagonal (angular velocity)
        msg.angular_velocity_covariance[8] = 0.01 * 0.01;  // σ² for z-axis

        imu_pub_->publish(msg);
    }

    // ── Odometry ──────────────────────────────────────────────────────────────
    void publishOdom()
    {
        auto gt = groundTruth(t_);

        nav_msgs::msg::Odometry msg;
        msg.header.stamp    = now();
        msg.header.frame_id = "odom";
        msg.child_frame_id  = "base_link";

        msg.twist.twist.linear.x = gt.v + odom_noise_(gen_);

        // Populate covariance so the EKF can optionally use it
        msg.twist.covariance[0] = 0.05 * 0.05;

        odom_pub_->publish(msg);
    }

    // ── Pose (GPS / VIO) ──────────────────────────────────────────────────────
    void publishPose()
    {
        auto gt = groundTruth(t_);

        geometry_msgs::msg::PoseStamped msg;
        msg.header.stamp    = now();
        msg.header.frame_id = "map";

        msg.pose.position.x = gt.x + pose_noise_(gen_);
        msg.pose.position.y = gt.y + pose_noise_(gen_);
        msg.pose.position.z = 0.0;

        pose_pub_->publish(msg);

        RCLCPP_DEBUG(get_logger(),
            "GT  x=%.2f y=%.2f | Meas x=%.2f y=%.2f",
            gt.x, gt.y, msg.pose.position.x, msg.pose.position.y);
    }

    // ── State ──────────────────────────────────────────────────────────────────
    double t_;
    std::mt19937                         gen_;
    mutable std::normal_distribution<double> imu_noise_;
    mutable std::normal_distribution<double> odom_noise_;
    mutable std::normal_distribution<double> pose_noise_;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr        odom_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr          imu_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;

    rclcpp::TimerBase::SharedPtr imu_timer_;
    rclcpp::TimerBase::SharedPtr odom_timer_;
    rclcpp::TimerBase::SharedPtr pose_timer_;
};

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MockSensorPublisher>());
    rclcpp::shutdown();
    return 0;
}