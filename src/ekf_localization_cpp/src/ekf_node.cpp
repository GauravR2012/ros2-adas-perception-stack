#include "ekf_localization_cpp/ekf_node.hpp"

EKFNode::EKFNode() : Node("ekf_localization_cpp")
{
    state_.setZero();
    covariance_.setIdentity();
    F_.setIdentity();
    Q_.setIdentity();

    RCLCPP_INFO(this->get_logger(), "EKF Localization Node Initialized");
}