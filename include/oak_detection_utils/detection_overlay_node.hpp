#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <mutex>

namespace oak_detection_utils {

class DetectionOverlayNode : public rclcpp::Node {
public:
  explicit DetectionOverlayNode(const rclcpp::NodeOptions & options);

private:
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image,
    vision_msgs::msg::Detection2DArray>;

  void sync_callback(
    const sensor_msgs::msg::Image::ConstSharedPtr & img_msg,
    const vision_msgs::msg::Detection2DArray::ConstSharedPtr & det_msg);
  void timer_callback();

  message_filters::Subscriber<sensor_msgs::msg::Image> img_sub_;
  message_filters::Subscriber<vision_msgs::msg::Detection2DArray> det_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::mutex cache_mutex_;
  cv::Mat cached_frame_;
  vision_msgs::msg::Detection2DArray::ConstSharedPtr cached_detections_;
  std_msgs::msg::Header cached_header_;

  std::vector<std::string> label_map_;
  int input_size_;
  bool show_dead_zone_;
};

}  // namespace oak_detection_utils
