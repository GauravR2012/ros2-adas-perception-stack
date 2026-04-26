#include <cmath>
#include "ekf_localization_cpp/ekf_node.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
EKFNode::EKFNode()
: Node("ekf_localization_cpp"),
  yaw_rate_(0.0),
  initialized_(false)
{
    // ── Declare tunable parameters ────────────────────────────────────────────
    declare_parameter("q_xy",   0.0002);
    declare_parameter("q_yaw",  0.00002);
    declare_parameter("q_vel",  0.002);
    declare_parameter("r_pose", 0.04);
    declare_parameter("r_vel",  0.0025);

    // ── Pose-driven mode (for nuScenes / offline datasets) ────────────────────
    // When true: prediction is driven by a fixed-rate timer instead of odom.
    // Odom is still used for velocity updates but does NOT trigger predict().
    // Use this when your data source is a dataset replayer, not live sensors.
    declare_parameter("pose_driven_mode", false);
    declare_parameter("predict_rate_hz",  10.0);

    // ── Build Q from parameters ───────────────────────────────────────────────
    Q_.setZero();
    Q_(0, 0) = get_parameter("q_xy").as_double();
    Q_(1, 1) = get_parameter("q_xy").as_double();
    Q_(2, 2) = get_parameter("q_yaw").as_double();
    Q_(3, 3) = get_parameter("q_vel").as_double();

    // ── Initialise state ──────────────────────────────────────────────────────
    state_.setZero();
    F_.setIdentity();

    // Start with physically meaningful uncertainty.
    // ±1 m position, ±18° yaw, ±0.5 m/s velocity.
    covariance_.setZero();
    covariance_(0, 0) = 1.0;
    covariance_(1, 1) = 1.0;
    covariance_(2, 2) = 0.1;
    covariance_(3, 3) = 0.5;

    // ── Subscriptions ─────────────────────────────────────────────────────────
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/odom", 10,
        std::bind(&EKFNode::odomCallback, this, std::placeholders::_1));

    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "/imu", 10,
        std::bind(&EKFNode::imuCallback, this, std::placeholders::_1));

    pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/pose_measurement", 10,
        std::bind(&EKFNode::poseCallback, this, std::placeholders::_1));

    // ── Publishers ────────────────────────────────────────────────────────────
    odom_pub_   = create_publisher<nav_msgs::msg::Odometry>("/ekf/odom", 10);
    marker_pub_ = create_publisher<visualization_msgs::msg::Marker>("/ekf/trajectory", 10);

    // ── TF broadcaster ────────────────────────────────────────────────────────
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // ── Pose-driven predict timer (nuScenes mode) ─────────────────────────────
    pose_driven_mode_ = get_parameter("pose_driven_mode").as_bool();

    if (pose_driven_mode_) {
        const double rate = get_parameter("predict_rate_hz").as_double();
        predict_timer_ = create_wall_timer(
            std::chrono::duration<double>(1.0 / rate),
            [this]() {
                if (!initialized_) return;
                auto now = this->now();
                double dt = (now - last_time_).seconds();
                last_time_ = now;
                if (dt <= 0.0 || dt > 1.0) return;
                predict(dt);
                publishOdometry();
            });
        RCLCPP_INFO(get_logger(),
            "Pose-driven mode ENABLED — predict timer at %.1f Hz. "
            "Odom used for velocity updates only.",
            get_parameter("predict_rate_hz").as_double());
    }

    // ── Watchdog: warn if odometry goes silent ────────────────────────────────
    watchdog_timer_ = create_wall_timer(
        std::chrono::milliseconds(500),
        std::bind(&EKFNode::watchdogCallback, this));

    last_time_ = this->now();

    RCLCPP_INFO(get_logger(),
        "EKF node started. Q_xy=%.4f Q_yaw=%.5f Q_vel=%.4f R_pose=%.4f R_vel=%.4f",
        Q_(0,0), Q_(2,2), Q_(3,3),
        get_parameter("r_pose").as_double(),
        get_parameter("r_vel").as_double());
}

// ─────────────────────────────────────────────────────────────────────────────
// EKF — Prediction step
// ─────────────────────────────────────────────────────────────────────────────
void EKFNode::predict(double dt)
{
    const double x   = state_(0);
    const double y   = state_(1);
    const double yaw = state_(2);
    const double v   = state_(3);

    // Integrate position with CURRENT yaw before advancing yaw.
    // Jacobian F_ uses pre-update yaw — must be consistent.
    state_(0) = x   + v * std::cos(yaw) * dt;
    state_(1) = y   + v * std::sin(yaw) * dt;
    state_(2) = yaw + yaw_rate_ * dt;
    state_(3) = v;

    normalizeYaw();

    // Jacobian of motion model (evaluated at pre-update yaw)
    F_.setIdentity();
    F_(0, 2) = -v * std::sin(yaw) * dt;
    F_(0, 3) =  std::cos(yaw) * dt;
    F_(1, 2) =  v * std::cos(yaw) * dt;
    F_(1, 3) =  std::sin(yaw) * dt;

    // Re-read Q from params every step so ros2 param set takes effect live.
    // Floor at 1e-6 to prevent covariance collapse.
    Q_(0,0) = std::max(get_parameter("q_xy").as_double(),  1e-6);
    Q_(1,1) = std::max(get_parameter("q_xy").as_double(),  1e-6);
    Q_(2,2) = std::max(get_parameter("q_yaw").as_double(), 1e-6);
    Q_(3,3) = std::max(get_parameter("q_vel").as_double(), 1e-6);

    // Covariance propagation — Q scaled by dt (rate-independent)
    covariance_ = F_ * covariance_ * F_.transpose() + Q_ * dt;
}

// ─────────────────────────────────────────────────────────────────────────────
// EKF — Pose measurement update  (x, y from PoseStamped)
// ─────────────────────────────────────────────────────────────────────────────
void EKFNode::updatePose(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    const double r_pose = get_parameter("r_pose").as_double();

    Eigen::Vector2d z;
    z << msg->pose.position.x, msg->pose.position.y;

    Eigen::Matrix<double, 2, 4> H;
    H.setZero();
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;

    Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * r_pose;

    Eigen::Vector2d innovation = z - H * state_;
    Eigen::Matrix2d S = H * covariance_ * H.transpose() + R;

    // NIS: should average ~2.0 for 2D measurement when well-tuned.
    // Consistently > 2 → R too small or Q too small.
    // Consistently < 2 → R too large or Q too large.
    double nis = innovation.transpose() * S.inverse() * innovation;
    RCLCPP_DEBUG(get_logger(), "Pose NIS: %.3f (target ~2.0)", nis);

    Eigen::Matrix<double, 4, 2> K = covariance_ * H.transpose() * S.inverse();

    state_ = state_ + K * innovation;

    // Joseph stabilised covariance update — preserves symmetry and
    // positive-definiteness under floating-point errors.
    Eigen::Matrix4d IKH = Eigen::Matrix4d::Identity() - K * H;
    covariance_ = IKH * covariance_ * IKH.transpose() + K * R * K.transpose();

    normalizeYaw();
}

// ─────────────────────────────────────────────────────────────────────────────
// EKF — Velocity measurement update  (from odometry twist)
// ─────────────────────────────────────────────────────────────────────────────
void EKFNode::updateVelocity(double v_measured)
{
    const double r_vel = get_parameter("r_vel").as_double();

    Eigen::Matrix<double, 1, 4> H_v;
    H_v << 0.0, 0.0, 0.0, 1.0;

    double innovation = v_measured - (H_v * state_)(0);
    double S          = (H_v * covariance_ * H_v.transpose())(0, 0) + r_vel;

    // NIS: should average ~1.0 for scalar measurement when well-tuned.
    RCLCPP_DEBUG(get_logger(), "Velocity NIS: %.3f (target ~1.0)",
        (innovation * innovation) / S);

    Eigen::Matrix<double, 4, 1> K_v = covariance_ * H_v.transpose() / S;

    state_ = state_ + K_v * innovation;

    // Joseph stabilised form for 1-D measurement
    Eigen::Matrix4d IKH = Eigen::Matrix4d::Identity() - K_v * H_v;
    covariance_ = IKH * covariance_ * IKH.transpose()
                + K_v * r_vel * K_v.transpose();
}

// ─────────────────────────────────────────────────────────────────────────────
// Callbacks
// ─────────────────────────────────────────────────────────────────────────────
void EKFNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    if (pose_driven_mode_) {
        // Pose-driven mode: predict timer drives the filter.
        // Odom only updates velocity state — no predict triggered here.
        if (!initialized_) return;
        updateVelocity(msg->twist.twist.linear.x);
        return;
    }

    // ── Standard odom-driven mode (mock sensors / real robot) ─────────────────
    auto current_time = this->now();

    if (!initialized_) {
        // Wait for first pose to initialize position before predicting.
        last_time_ = current_time;
        return;
    }

    double dt = (current_time - last_time_).seconds();
    last_time_ = current_time;

    if (dt <= 0.0 || dt > 1.0) {
        RCLCPP_WARN(get_logger(), "Skipping predict: bad dt=%.4f", dt);
        return;
    }

    predict(dt);
    updateVelocity(msg->twist.twist.linear.x);
    publishOdometry();
}

void EKFNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    // Store yaw rate — consumed by predict() on next tick.
    yaw_rate_ = msg->angular_velocity.z;
}

void EKFNode::poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (!initialized_) {
        // Initialize EKF state from first pose measurement.
        // Prevents enormous NIS spike when filter starts at (0,0)
        // but real-world data is at e.g. (411, 1180) in nuScenes.
        state_(0) = msg->pose.position.x;
        state_(1) = msg->pose.position.y;

        const double qx = msg->pose.orientation.x;
        const double qy = msg->pose.orientation.y;
        const double qz = msg->pose.orientation.z;
        const double qw = msg->pose.orientation.w;
        state_(2) = std::atan2(2.0 * (qw*qz + qx*qy),
                               1.0 - 2.0 * (qy*qy + qz*qz));
        state_(3) = 0.0;

        last_time_   = this->now();

        // ✅ ADD THIS LINE HERE
        trajectory_marker_.points.clear();

        initialized_ = true;

        RCLCPP_INFO(get_logger(),
            "EKF initialized from first pose: x=%.2f y=%.2f yaw=%.3f",
            state_(0), state_(1), state_(2));
        return;
    }

    updatePose(msg);

    // In odom-driven mode publish after each pose correction.
    // In pose-driven mode the predict timer handles publishing.
    if (!pose_driven_mode_) {
        publishOdometry();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Health watchdog
// ─────────────────────────────────────────────────────────────────────────────
void EKFNode::watchdogCallback()
{
    if (!initialized_) return;

    double age = (this->now() - last_time_).seconds();
    if (age > 0.5) {
        RCLCPP_WARN(get_logger(),
            "No odometry for %.2f s — dead reckoning may be unreliable.", age);
        // Q inflation removed — corrupts filter state across long outages.
        // Restart the node if sensor dropout is sustained.
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
void EKFNode::normalizeYaw()
{
    state_(2) = std::atan2(std::sin(state_(2)), std::cos(state_(2)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Publishing
// ─────────────────────────────────────────────────────────────────────────────
void EKFNode::publishOdometry()
{
    const auto stamp = this->now();

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, state_(2));

    // ── Odometry message ──────────────────────────────────────────────────────
    nav_msgs::msg::Odometry odom_msg;
    odom_msg.header.stamp    = stamp;
    odom_msg.header.frame_id = "odom";
    odom_msg.child_frame_id  = "base_link";

    odom_msg.pose.pose.position.x    = state_(0);
    odom_msg.pose.pose.position.y    = state_(1);
    odom_msg.pose.pose.position.z    = 0.0;
    odom_msg.pose.pose.orientation.x = q.x();
    odom_msg.pose.pose.orientation.y = q.y();
    odom_msg.pose.pose.orientation.z = q.z();
    odom_msg.pose.pose.orientation.w = q.w();
    odom_msg.twist.twist.linear.x    = state_(3);

    // =======================
    // Pose covariance (6x6)
    // =======================

    // Reset everything first
    odom_msg.pose.covariance.fill(0.0);

    // x, y from EKF covariance
    odom_msg.pose.covariance[0] = covariance_(0, 0);   // x-x
    odom_msg.pose.covariance[1] = covariance_(0, 1);   // x-y
    odom_msg.pose.covariance[6] = covariance_(1, 0);   // y-x
    odom_msg.pose.covariance[7] = covariance_(1, 1);   // y-y

    // yaw from EKF
    odom_msg.pose.covariance[35] = covariance_(2, 2);  // yaw-yaw

    // Fill unused dimensions with small noise (VERY IMPORTANT)
    odom_msg.pose.covariance[14] = 0.1;  // z-z
    odom_msg.pose.covariance[21] = 0.1;  // roll-roll
    odom_msg.pose.covariance[28] = 0.1;  // pitch-pitch


    // =======================
    // Twist covariance (6x6)
    // =======================

    odom_msg.twist.covariance.fill(0.0);

    // linear velocity uncertainty
    odom_msg.twist.covariance[0] = 0.1;   // vx
    odom_msg.twist.covariance[7] = 0.1;   // vy

    // angular velocity uncertainty
    odom_msg.twist.covariance[35] = 0.2;  // yaw rate

    odom_pub_->publish(odom_msg);

    // ── TF broadcast (required by RViz, Nav2, SLAM) ───────────────────────────
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp            = stamp;
    tf_msg.header.frame_id         = "odom";
    tf_msg.child_frame_id          = "base_link";
    tf_msg.transform.translation.x = state_(0);
    tf_msg.transform.translation.y = state_(1);
    tf_msg.transform.translation.z = 0.0;
    tf_msg.transform.rotation      = odom_msg.pose.pose.orientation;
    tf_broadcaster_->sendTransform(tf_msg);

    // ── Trajectory marker for RViz ────────────────────────────────────────────
    publishTrajectoryMarker();
}

void EKFNode::publishTrajectoryMarker()
{
    const auto stamp = this->now();

    trajectory_marker_.header.stamp    = stamp;
    trajectory_marker_.header.frame_id = "odom";
    trajectory_marker_.ns              = "ekf_trajectory";
    trajectory_marker_.id              = 0;
    trajectory_marker_.type            = visualization_msgs::msg::Marker::LINE_STRIP;
    trajectory_marker_.action          = visualization_msgs::msg::Marker::ADD;
    trajectory_marker_.scale.x         = 0.3;   // wider for real-world scale (metres)

    // Cyan trail
    trajectory_marker_.color.r = 0.0f;
    trajectory_marker_.color.g = 0.8f;
    trajectory_marker_.color.b = 1.0f;
    trajectory_marker_.color.a = 1.0f;

    geometry_msgs::msg::Point p;
    p.x = state_(0);
    p.y = state_(1);
    p.z = 0.0;
    trajectory_marker_.points.push_back(p);

    // Cap history to avoid unbounded memory growth
    if (trajectory_marker_.points.size() > 2000) {
        trajectory_marker_.points.erase(trajectory_marker_.points.begin());
    }

    marker_pub_->publish(trajectory_marker_);
}