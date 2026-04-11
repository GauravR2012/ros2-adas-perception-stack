#pragma once

#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>

class EKFNode : public rclcpp::Node
{
public:
    EKFNode();

private:
    Eigen::Vector4d state_;
    Eigen::Matrix4d covariance_;
    Eigen::Matrix4d F_;
    Eigen::Matrix4d Q_;
};   