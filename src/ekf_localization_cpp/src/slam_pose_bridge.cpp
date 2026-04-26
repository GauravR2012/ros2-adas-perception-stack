#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

class SlamPoseBridge : public rclcpp::Node
{
public:
    SlamPoseBridge() : Node("slam_pose_bridge")
    {
        tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
            "/pose_measurement", 10);

        // Poll TF at 10 Hz — matches slam_toolbox update rate
        timer_ = create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&SlamPoseBridge::publishPose, this));

        RCLCPP_INFO(get_logger(), "SLAM pose bridge started — map→base_link → /pose_measurement");
    }

private:
    void publishPose()
    {
        geometry_msgs::msg::TransformStamped tf_msg;

        try {
            // Get full robot pose in map frame
            tf_msg = tf_buffer_->lookupTransform(
                "map", "base_link",
                tf2::TimePointZero);
        }
        catch (tf2::TransformException & ex) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "Waiting for map→base_link TF: %s", ex.what());
            return;
        }

        // Convert TF to PoseStamped for EKF
        geometry_msgs::msg::PoseStamped pose_msg;
        pose_msg.header.stamp    = tf_msg.header.stamp;
        pose_msg.header.frame_id = "map";
        pose_msg.pose.position.x = tf_msg.transform.translation.x;
        pose_msg.pose.position.y = tf_msg.transform.translation.y;
        pose_msg.pose.position.z = 0.0;
        pose_msg.pose.orientation = tf_msg.transform.rotation;

        pose_pub_->publish(pose_msg);
    }

    std::shared_ptr<tf2_ros::Buffer>            tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SlamPoseBridge>());
    rclcpp::shutdown();
    return 0;
}   