#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <cmath>
#include <limits>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
// PointCloud2 → LaserScan converter
//
// Subscribes to a 3D PointCloud2 topic and projects it into a 2D LaserScan
// by filtering points within a configurable height band [min_z, max_z].
// This is required because slam_toolbox expects a LaserScan, not a PointCloud2.
// ─────────────────────────────────────────────────────────────────────────────

class PointCloudToLaserScan : public rclcpp::Node
{
public:
    PointCloudToLaserScan() : Node("pointcloud_to_laserscan")
    {
        // ── Parameters ────────────────────────────────────────────────────────
        declare_parameter("min_z", -0.3);          // lower height bound (m)
        declare_parameter("max_z",  1.5);           // upper height bound (m)
        declare_parameter("range_min", 0.3);        // minimum valid range (m)
        declare_parameter("range_max", 50.0);       // maximum valid range (m)
        declare_parameter("angle_min", -M_PI);      // scan start angle (rad)
        declare_parameter("angle_max",  M_PI);      // scan end angle (rad)
        declare_parameter("angle_increment", 0.00436332); // ~0.25° resolution
        declare_parameter("scan_frame", std::string("lidar"));
        declare_parameter("input_topic", std::string("/lidar/points"));
        declare_parameter("output_topic", std::string("/scan"));

        min_z_            = get_parameter("min_z").as_double();
        max_z_            = get_parameter("max_z").as_double();
        range_min_        = get_parameter("range_min").as_double();
        range_max_        = get_parameter("range_max").as_double();
        angle_min_        = get_parameter("angle_min").as_double();
        angle_max_        = get_parameter("angle_max").as_double();
        angle_increment_  = get_parameter("angle_increment").as_double();
        scan_frame_       = get_parameter("scan_frame").as_string();

        std::string input_topic  = get_parameter("input_topic").as_string();
        std::string output_topic = get_parameter("output_topic").as_string();

        // ── Subscriber ────────────────────────────────────────────────────────
        pc_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic, rclcpp::SensorDataQoS(),
            std::bind(&PointCloudToLaserScan::cloudCallback, this,
                       std::placeholders::_1));

        // ── Publisher ─────────────────────────────────────────────────────────
        scan_pub_ = create_publisher<sensor_msgs::msg::LaserScan>(
            output_topic, 10);

        RCLCPP_INFO(get_logger(),
            "PointCloud→LaserScan started: %s → %s  "
            "z∈[%.2f, %.2f]  range∈[%.1f, %.1f]  frame=%s",
            input_topic.c_str(), output_topic.c_str(),
            min_z_, max_z_, range_min_, range_max_, scan_frame_.c_str());
    }

private:
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // ── Build empty LaserScan ─────────────────────────────────────────────
        auto scan = sensor_msgs::msg::LaserScan();
        scan.header.stamp    = msg->header.stamp;
        scan.header.frame_id = scan_frame_;

        scan.angle_min       = angle_min_;
        scan.angle_max       = angle_max_;
        scan.angle_increment = angle_increment_;
        scan.range_min       = range_min_;
        scan.range_max       = range_max_;
        scan.time_increment  = 0.0f;
        scan.scan_time       = 0.1f;

        const int num_bins = static_cast<int>(
            std::ceil((angle_max_ - angle_min_) / angle_increment_));
        scan.ranges.assign(num_bins, std::numeric_limits<float>::infinity());

        // ── Iterate over point cloud ──────────────────────────────────────────
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
        {
            const float x = *iter_x;
            const float y = *iter_y;
            const float z = *iter_z;

            // Height-band filter
            if (z < min_z_ || z > max_z_) continue;

            // Compute range in XY plane
            const float range = std::sqrt(x * x + y * y);
            if (range < range_min_ || range > range_max_) continue;

            // Compute angle
            const float angle = std::atan2(y, x);
            if (angle < angle_min_ || angle > angle_max_) continue;

            // Find bin index
            const int idx = static_cast<int>(
                (angle - angle_min_) / angle_increment_);
            if (idx < 0 || idx >= num_bins) continue;

            // Keep minimum range per bin (closest obstacle wins)
            if (range < scan.ranges[idx]) {
                scan.ranges[idx] = range;
            }
        }

        scan_pub_->publish(scan);
    }

    // ── Members ───────────────────────────────────────────────────────────────
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr scan_pub_;

    double min_z_, max_z_;
    double range_min_, range_max_;
    double angle_min_, angle_max_;
    double angle_increment_;
    std::string scan_frame_;
};

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudToLaserScan>());
    rclcpp::shutdown();
    return 0;
}
