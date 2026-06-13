#pragma once

#include <rclcpp/rclcpp.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <Eigen/Dense>
#include <memory>

class EKFNode : public rclcpp::Node
{
public:
    EKFNode();

private:
    // ── Core EKF steps ────────────────────────────────────────────────────────
    void predict(double dt);
    void updatePose(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void updateVelocity(double v_measured);

    // ── ROS callbacks ─────────────────────────────────────────────────────────
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

    // ── Publishing ────────────────────────────────────────────────────────────
    void publishOdometry(const builtin_interfaces::msg::Time & stamp);
    void publishTrajectoryMarker(const builtin_interfaces::msg::Time & stamp);

    // ── Health monitoring ─────────────────────────────────────────────────────
    void watchdogCallback();
    void normalizeYaw();

    // ── Subscriptions ─────────────────────────────────────────────────────────
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr         odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr           imu_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;

    // ── Publishers ────────────────────────────────────────────────────────────
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr          odom_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr  marker_pub_;

    // ── TF ────────────────────────────────────────────────────────────────────
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // ── Timers ────────────────────────────────────────────────────────────────
    rclcpp::TimerBase::SharedPtr watchdog_timer_;

    // Pose-driven mode: replaces odom-triggered predict with a fixed-rate timer.
    // Used when replaying offline datasets (e.g. nuScenes) where odom arrives
    // at the same low rate as pose and cannot drive the filter at high frequency.
    rclcpp::TimerBase::SharedPtr predict_timer_;
    bool pose_driven_mode_;

    // ── EKF state ─────────────────────────────────────────────────────────────
    // x = [x, y, yaw, v]^T
    Eigen::Vector4d state_;
    Eigen::Matrix4d covariance_;
    Eigen::Matrix4d F_;   // Jacobian of motion model
    Eigen::Matrix4d Q_;   // Process noise covariance

    // ── IMU data ──────────────────────────────────────────────────────────────
    double yaw_rate_;

    // ── Timing ────────────────────────────────────────────────────────────────
    rclcpp::Time last_time_;
    bool initialized_;

    // ── Visualization ─────────────────────────────────────────────────────────
    visualization_msgs::msg::Marker trajectory_marker_;
};