#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <cmath>

class FakeLaser : public rclcpp::Node
{
public:
    FakeLaser() : Node("fake_laser")
    {
        pub_ = create_publisher<sensor_msgs::msg::LaserScan>("/scan", 10);
        timer_ = create_wall_timer(std::chrono::milliseconds(100),
            std::bind(&FakeLaser::publish, this));
    }

private:
    void publish()
    {
        sensor_msgs::msg::LaserScan msg;
        msg.header.stamp    = now();
        msg.header.frame_id = "base_link";
        msg.angle_min       = -M_PI;
        msg.angle_max       =  M_PI;
        msg.angle_increment =  M_PI / 180.0;  // 1 degree resolution
        msg.range_min       = 0.1f;
        msg.range_max       = 10.0f;
        msg.time_increment  = 0.0f;
        msg.scan_time       = 0.1f;

        // Fill with a constant range — simulates open room walls
        int num_readings = static_cast<int>(
            (msg.angle_max - msg.angle_min) / msg.angle_increment);
        msg.ranges.assign(num_readings, 3.0f);

        pub_->publish(msg);
    }

    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FakeLaser>());
    rclcpp::shutdown();
    return 0;
}