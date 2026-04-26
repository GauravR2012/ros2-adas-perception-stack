cmake_minimum_required(VERSION 3.8)
project(ekf_localization_cpp)

# ── Compiler flags ─────────────────────────────────────────────────────────────
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

# ── Dependencies ───────────────────────────────────────────────────────────────
find_package(ament_cmake          REQUIRED)
find_package(rclcpp               REQUIRED)
find_package(nav_msgs             REQUIRED)
find_package(sensor_msgs          REQUIRED)
find_package(geometry_msgs        REQUIRED)
find_package(visualization_msgs   REQUIRED)   # NEW: trajectory marker
find_package(tf2_ros              REQUIRED)   # NEW: TF broadcast
find_package(tf2                  REQUIRED)   # NEW: tf2::Quaternion
find_package(Eigen3               REQUIRED)

# ── EKF node library ──────────────────────────────────────────────────────────
add_library(ekf_node_lib
  src/ekf_node.cpp
)

target_include_directories(ekf_node_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
)

ament_target_dependencies(ekf_node_lib
  rclcpp
  nav_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  tf2_ros
  tf2
)

# ── EKF executable ────────────────────────────────────────────────────────────
add_executable(ekf_node
  src/main.cpp
)

target_link_libraries(ekf_node ekf_node_lib)

ament_target_dependencies(ekf_node
  rclcpp
)

# ── Mock sensor executable ────────────────────────────────────────────────────
# Compile with:  colcon build --packages-select ekf_localization_cpp
# Run with:      ros2 run ekf_localization_cpp mock_sensors
add_executable(mock_sensors
  src/mock_sensors.cpp
)

ament_target_dependencies(mock_sensors
  rclcpp
  nav_msgs
  sensor_msgs
  geometry_msgs
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS
  ekf_node_lib
  ekf_node
  mock_sensors
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)



# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

find_package(tf2_geometry_msgs REQUIRED)

add_executable(slam_pose_bridge src/slam_pose_bridge.cpp)
ament_target_dependencies(slam_pose_bridge
  rclcpp geometry_msgs tf2_ros tf2_geometry_msgs)

install(TARGETS slam_pose_bridge
  DESTINATION lib/${PROJECT_NAME})

ament_package()