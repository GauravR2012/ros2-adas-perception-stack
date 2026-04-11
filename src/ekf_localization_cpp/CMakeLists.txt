cmake_minimum_required(VERSION 3.8)
project(ekf_localization_cpp)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(ekf_node
  src/main.cpp
  src/ekf_node.cpp
)

ament_target_dependencies(ekf_node rclcpp)

target_link_libraries(ekf_node Eigen3::Eigen)

install(TARGETS
  ekf_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
