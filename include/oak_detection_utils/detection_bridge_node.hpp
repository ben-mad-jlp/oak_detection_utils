#pragma once

#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <depthai_yolo_msgs/msg/bounding_boxes.hpp>

namespace oak_detection_utils {

class DetectionBridgeNode : public rclcpp::Node {
public:
  explicit DetectionBridgeNode(const rclcpp::NodeOptions & options);

private:
  void detection_callback(const vision_msgs::msg::Detection2DArray::SharedPtr msg);

  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr sub_;
  rclcpp::Publisher<depthai_yolo_msgs::msg::BoundingBoxes>::SharedPtr pub_;

  std::vector<std::string> label_map_;
  int input_size_;
};

}  // namespace oak_detection_utils
