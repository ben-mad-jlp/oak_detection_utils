#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

namespace oak_detection_utils {

enum class TrackState { TENTATIVE, CONFIRMED, LOST };

struct TrackedObject {
  int track_id;
  int class_id;
  std::string class_name;
  double confidence;
  double xmin, ymin, xmax, ymax;
  int frames_seen;
  int frames_missing;
  TrackState state;
};

class DetectionCaptureNode : public rclcpp::Node {
public:
  explicit DetectionCaptureNode(const rclcpp::NodeOptions & options);

private:
  void detection_callback(const vision_msgs::msg::Detection2DArray::SharedPtr msg);
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  void periodic_timer_callback();
  void diagnostics_timer_callback();

  void enable_callback(
    const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
    std::shared_ptr<std_srvs::srv::SetBool::Response> res);
  void start_session_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res);
  void stop_session_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
    std::shared_ptr<std_srvs::srv::Trigger::Response> res);

  void update_tracks(
    const std::vector<vision_msgs::msg::Detection2D> & detections);
  void save_capture(const std::string & reason);
  double compute_iou(const TrackedObject & t,
    double xmin, double ymin, double xmax, double ymax) const;
  std::string get_session_dir() const;
  uint64_t compute_disk_usage(const std::string & path) const;

  // Subscriptions
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr det_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;

  // Services
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_session_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stop_session_srv_;

  // Publisher
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr diag_pub_;

  // Timers
  rclcpp::TimerBase::SharedPtr periodic_timer_;
  rclcpp::TimerBase::SharedPtr diagnostics_timer_;

  // State
  bool enabled_;
  bool session_active_;
  std::vector<TrackedObject> tracks_;
  int next_track_id_;
  cv::Mat latest_image_;
  std_msgs::msg::Header latest_image_header_;
  rclcpp::Time last_save_time_;
  int total_saves_;
  int session_saves_;
  std::atomic<uint64_t> disk_usage_bytes_{0};

  // Mutexes
  std::mutex image_mutex_;
  std::mutex track_mutex_;
  std::mutex save_mutex_;
  std::mutex session_mutex_;

  // Parameters
  std::string output_dir_;
  std::string camera_name_;
  std::string session_name_;
  int confirm_frames_;
  int lost_frames_;
  double iou_threshold_;
  double periodic_interval_;
  double min_save_interval_;
  bool save_negatives_;
  int64_t max_disk_usage_mb_;
  int jpeg_quality_;
  std::vector<std::string> label_map_;
};

}  // namespace oak_detection_utils
