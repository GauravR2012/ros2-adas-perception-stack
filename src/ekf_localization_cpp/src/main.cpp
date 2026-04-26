#include <rclcpp/rclcpp.hpp>
#include "ekf_localization_cpp/ekf_node.hpp"

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<EKFNode>();
    RCLCPP_INFO(node->get_logger(), "Spinning EKF localization node...");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}