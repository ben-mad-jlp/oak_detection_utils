cmake_minimum_required(VERSION 3.8)
project(oak_detection_utils)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(depthai_yolo_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)

# Detection Bridge Node
add_library(detection_bridge_node SHARED
  src/detection_bridge_node.cpp)
target_include_directories(detection_bridge_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(detection_bridge_node
  rclcpp rclcpp_components vision_msgs depthai_yolo_msgs std_msgs)
rclcpp_components_register_nodes(detection_bridge_node
  "oak_detection_utils::DetectionBridgeNode")

# Detection Overlay Node
add_library(detection_overlay_node SHARED
  src/detection_overlay_node.cpp)
target_include_directories(detection_overlay_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(detection_overlay_node ${OpenCV_LIBS})
ament_target_dependencies(detection_overlay_node
  rclcpp rclcpp_components vision_msgs sensor_msgs std_msgs
  cv_bridge message_filters)
rclcpp_components_register_nodes(detection_overlay_node
  "oak_detection_utils::DetectionOverlayNode")

install(TARGETS detection_bridge_node detection_overlay_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(DIRECTORY include/
  DESTINATION include)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch)

ament_package()
