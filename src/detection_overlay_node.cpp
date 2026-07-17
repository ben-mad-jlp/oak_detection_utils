#include "oak_detection_utils/detection_overlay_node.hpp"

namespace oak_detection_utils {

DetectionOverlayNode::DetectionOverlayNode(const rclcpp::NodeOptions & options)
: Node("detection_overlay", options)
{
  declare_parameter<double>("publish_rate", 5.0);
  declare_parameter<int>("input_size", 416);
  declare_parameter<std::vector<std::string>>("label_map", std::vector<std::string>{});
  declare_parameter<bool>("show_dead_zone", true);

  input_size_ = get_parameter("input_size").as_int();
  label_map_ = get_parameter("label_map").as_string_array();
  show_dead_zone_ = get_parameter("show_dead_zone").as_bool();
  double rate = get_parameter("publish_rate").as_double();

  img_sub_.subscribe(this, "rgb/image_raw", rmw_qos_profile_default);
  det_sub_.subscribe(this, "nn/detections", rmw_qos_profile_default);

  sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
    SyncPolicy(10), img_sub_, det_sub_);
  sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.1));
  sync_->registerCallback(
    std::bind(&DetectionOverlayNode::sync_callback, this,
      std::placeholders::_1, std::placeholders::_2));

  pub_ = create_publisher<sensor_msgs::msg::Image>("~/color/overlay", 1);

  auto period = std::chrono::duration<double>(1.0 / rate);
  timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(period),
    std::bind(&DetectionOverlayNode::timer_callback, this));

  RCLCPP_INFO(get_logger(), "DetectionOverlayNode started (rate=%.1f Hz, input_size=%d)",
    rate, input_size_);
}

void DetectionOverlayNode::sync_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr & img_msg,
  const vision_msgs::msg::Detection2DArray::ConstSharedPtr & det_msg)
{
  try {
    auto cv_ptr = cv_bridge::toCvCopy(img_msg, "bgr8");
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cached_frame_ = cv_ptr->image;
    cached_detections_ = det_msg;
    cached_header_ = img_msg->header;
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
      "cv_bridge exception: %s", e.what());
  }
}

void DetectionOverlayNode::timer_callback()
{
  if (pub_->get_subscription_count() == 0) {
    return;
  }

  cv::Mat frame;
  vision_msgs::msg::Detection2DArray::ConstSharedPtr detections;
  std_msgs::msg::Header header;

  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (cached_frame_.empty()) {
      return;
    }
    frame = cached_frame_.clone();
    detections = cached_detections_;
    header = cached_header_;
  }

  int img_w = frame.cols;
  int img_h = frame.rows;

  // NN uses a square crop from the center of the image, so BOTH axes have a dead-zone
  // margin: the larger dimension is cropped. On a landscape frame that is the left/right
  // strips (offset_y = 0); on a portrait frame it is the top/bottom strips (offset_x = 0).
  int nn_region = std::min(img_w, img_h);
  int offset_x = (img_w - nn_region) / 2;
  int offset_y = (img_h - nn_region) / 2;

  // Draw dead zones (strips not seen by NN), on whichever axis is cropped.
  if (show_dead_zone_) {
    auto shade = [](cv::Mat roi) {
      cv::Mat gray_overlay(roi.size(), roi.type(), cv::Scalar(80, 80, 80));
      cv::addWeighted(roi, 0.4, gray_overlay, 0.6, 0, roi);
    };
    if (offset_x > 0) {
      shade(frame(cv::Rect(0, 0, offset_x, img_h)));
      shade(frame(cv::Rect(img_w - offset_x, 0, offset_x, img_h)));
    }
    if (offset_y > 0) {
      shade(frame(cv::Rect(0, 0, img_w, offset_y)));
      shade(frame(cv::Rect(0, img_h - offset_y, img_w, offset_y)));
    }
  }

  // Draw detections
  if (detections) {
    double scale = static_cast<double>(nn_region) / input_size_;

    for (const auto & det : detections->detections) {
      double cx = det.bbox.center.position.x;
      double cy = det.bbox.center.position.y;
      double w = det.bbox.size_x;
      double h = det.bbox.size_y;

      // Scale from NN coords to image coords, through the centre crop on both axes.
      int x1 = static_cast<int>((cx - w / 2.0) * scale) + offset_x;
      int y1 = static_cast<int>((cy - h / 2.0) * scale) + offset_y;
      int x2 = static_cast<int>((cx + w / 2.0) * scale) + offset_x;
      int y2 = static_cast<int>((cy + h / 2.0) * scale) + offset_y;

      cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2),
        cv::Scalar(0, 255, 0), 2);

      // Build label text
      std::string label;
      double conf = 0.0;
      if (!det.results.empty()) {
        conf = det.results[0].hypothesis.score;
        const std::string & class_id_str = det.results[0].hypothesis.class_id;
        // class_id may be either a numeric index (look up in label_map) or
        // an already-resolved class name (use verbatim). atoi-ing a string
        // name silently returns 0, mislabeling every detection.
        char * end = nullptr;
        long class_id = std::strtol(class_id_str.c_str(), &end, 10);
        bool is_numeric = !class_id_str.empty() && *end == '\0';
        if (is_numeric && class_id >= 0 && class_id < static_cast<long>(label_map_.size())) {
          label = label_map_[class_id];
        } else {
          label = class_id_str;
        }
      }
      char buf[64];
      std::snprintf(buf, sizeof(buf), "%s %.0f%%", label.c_str(), conf * 100.0);

      int baseline = 0;
      cv::Size text_size = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
      cv::rectangle(frame,
        cv::Point(x1, y1 - text_size.height - 4),
        cv::Point(x1 + text_size.width, y1),
        cv::Scalar(0, 255, 0), cv::FILLED);
      cv::putText(frame, buf, cv::Point(x1, y1 - 2),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
  }

  auto out_msg = cv_bridge::CvImage(header, "bgr8", frame).toImageMsg();
  pub_->publish(*out_msg);
}

}  // namespace oak_detection_utils

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(oak_detection_utils::DetectionOverlayNode)
