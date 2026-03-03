#include "oak_detection_utils/detection_bridge_node.hpp"

#include <cmath>

namespace oak_detection_utils {

DetectionBridgeNode::DetectionBridgeNode(const rclcpp::NodeOptions & options)
: Node("detection_bridge", options)
{
  declare_parameter<std::vector<std::string>>("label_map", std::vector<std::string>{});
  declare_parameter<int>("input_size", 416);
  declare_parameter<double>("confidence_threshold", 0.25);

  label_map_ = get_parameter("label_map").as_string_array();
  input_size_ = get_parameter("input_size").as_int();
  confidence_threshold_ = get_parameter("confidence_threshold").as_double();

  sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
    "/oak/nn/detections", 10,
    std::bind(&DetectionBridgeNode::detection_callback, this, std::placeholders::_1));

  pub_ = create_publisher<depthai_yolo_msgs::msg::BoundingBoxes>(
    "~/detections", rclcpp::SensorDataQoS());

  RCLCPP_INFO(get_logger(), "DetectionBridgeNode started (input_size=%d, %zu labels)",
    input_size_, label_map_.size());
}

void DetectionBridgeNode::detection_callback(
  const vision_msgs::msg::Detection2DArray::SharedPtr msg)
{
  depthai_yolo_msgs::msg::BoundingBoxes out;
  out.header = msg->header;
  out.image_header = msg->header;

  for (const auto & det : msg->detections) {
    depthai_yolo_msgs::msg::BoundingBox bb;

    // Detection2D.bbox uses center + size_x/size_y
    double cx = det.bbox.center.position.x;
    double cy = det.bbox.center.position.y;
    double w = det.bbox.size_x;
    double h = det.bbox.size_y;

    // The official driver outputs bbox in NN input pixels (input_size x input_size).
    // Convert center+size to corner coords as integers.
    bb.xmin = static_cast<int64_t>(cx - w / 2.0);
    bb.ymin = static_cast<int64_t>(cy - h / 2.0);
    bb.xmax = static_cast<int64_t>(cx + w / 2.0);
    bb.ymax = static_cast<int64_t>(cy + h / 2.0);

    // Clamp to input_size bounds
    bb.xmin = std::max(bb.xmin, static_cast<int64_t>(0));
    bb.ymin = std::max(bb.ymin, static_cast<int64_t>(0));
    bb.xmax = std::min(bb.xmax, static_cast<int64_t>(input_size_));
    bb.ymax = std::min(bb.ymax, static_cast<int64_t>(input_size_));

    if (!det.results.empty()) {
      double raw_score = det.results[0].hypothesis.score;
      // Apply sigmoid if score is outside [0, 1] (raw logits from parser)
      double prob = (raw_score < 0.0 || raw_score > 1.0)
        ? 1.0 / (1.0 + std::exp(-raw_score))
        : raw_score;

      if (prob < confidence_threshold_) {
        continue;
      }

      bb.probability = prob;
      int class_id = std::atoi(det.results[0].hypothesis.class_id.c_str());
      if (class_id >= 0 && class_id < static_cast<int>(label_map_.size())) {
        bb.class_name = label_map_[class_id];
      } else {
        bb.class_name = det.results[0].hypothesis.class_id;
      }
      bb.id = static_cast<int16_t>(class_id);
    }

    out.bounding_boxes.push_back(bb);
  }

  pub_->publish(out);
}

}  // namespace oak_detection_utils

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(oak_detection_utils::DetectionBridgeNode)
