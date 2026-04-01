#include "oak_detection_utils/detection_capture_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>

namespace fs = std::filesystem;

namespace oak_detection_utils {

DetectionCaptureNode::DetectionCaptureNode(const rclcpp::NodeOptions & options)
: Node("capture", options),
  enabled_(false),
  session_active_(false),
  next_track_id_(0),
  last_save_time_(0, 0, RCL_ROS_TIME),
  total_saves_(0),
  session_saves_(0)
{
  declare_parameter<bool>("enabled", false);
  declare_parameter<std::string>("output_dir", "~/capture");
  declare_parameter<std::string>("camera_name", "");
  declare_parameter<int>("confirm_frames", 5);
  declare_parameter<int>("lost_frames", 15);
  declare_parameter<double>("iou_threshold", 0.3);
  declare_parameter<double>("periodic_interval", 30.0);
  declare_parameter<double>("min_save_interval", 2.0);
  declare_parameter<bool>("save_negatives", false);
  declare_parameter<int>("max_disk_usage_mb", 2048);
  declare_parameter<int>("jpeg_quality", 85);
  declare_parameter<std::vector<std::string>>("label_map", std::vector<std::string>{});
  declare_parameter<std::string>("session_name", "default");

  enabled_ = get_parameter("enabled").as_bool();
  output_dir_ = get_parameter("output_dir").as_string();
  camera_name_ = get_parameter("camera_name").as_string();
  confirm_frames_ = get_parameter("confirm_frames").as_int();
  lost_frames_ = get_parameter("lost_frames").as_int();
  iou_threshold_ = get_parameter("iou_threshold").as_double();
  periodic_interval_ = get_parameter("periodic_interval").as_double();
  min_save_interval_ = get_parameter("min_save_interval").as_double();
  save_negatives_ = get_parameter("save_negatives").as_bool();
  max_disk_usage_mb_ = get_parameter("max_disk_usage_mb").as_int();
  jpeg_quality_ = get_parameter("jpeg_quality").as_int();
  label_map_ = get_parameter("label_map").as_string_array();
  session_name_ = get_parameter("session_name").as_string();

  // Expand ~ in output_dir
  if (!output_dir_.empty() && output_dir_[0] == '~') {
    const char * home = std::getenv("HOME");
    if (home) {
      output_dir_ = std::string(home) + output_dir_.substr(1);
    }
  }

  // Derive camera_name from namespace if empty
  if (camera_name_.empty()) {
    std::string ns = get_namespace();
    // Strip leading slashes
    while (!ns.empty() && ns[0] == '/') {
      ns = ns.substr(1);
    }
    camera_name_ = ns.empty() ? "unknown_camera" : ns;
    // Replace remaining slashes with underscores
    std::replace(camera_name_.begin(), camera_name_.end(), '/', '_');
  }

  // Always subscribe to detections
  det_sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
    "nn/detections", 10,
    std::bind(&DetectionCaptureNode::detection_callback, this,
      std::placeholders::_1));

  // Subscribe to image only if enabled
  if (enabled_) {
    img_sub_ = create_subscription<sensor_msgs::msg::Image>(
      "rgb/image_raw", rclcpp::SensorDataQoS(),
      std::bind(&DetectionCaptureNode::image_callback, this,
        std::placeholders::_1));
  }

  // Services
  enable_srv_ = create_service<std_srvs::srv::SetBool>(
    "~/enable",
    std::bind(&DetectionCaptureNode::enable_callback, this,
      std::placeholders::_1, std::placeholders::_2));
  start_session_srv_ = create_service<std_srvs::srv::Trigger>(
    "~/start_session",
    std::bind(&DetectionCaptureNode::start_session_callback, this,
      std::placeholders::_1, std::placeholders::_2));
  stop_session_srv_ = create_service<std_srvs::srv::Trigger>(
    "~/stop_session",
    std::bind(&DetectionCaptureNode::stop_session_callback, this,
      std::placeholders::_1, std::placeholders::_2));

  // Diagnostics publisher
  diag_pub_ = create_publisher<std_msgs::msg::String>("~/diagnostics", 10);

  // Timers
  periodic_timer_ = create_wall_timer(
    std::chrono::duration<double>(periodic_interval_),
    std::bind(&DetectionCaptureNode::periodic_timer_callback, this));
  diagnostics_timer_ = create_wall_timer(
    std::chrono::seconds(1),
    std::bind(&DetectionCaptureNode::diagnostics_timer_callback, this));

  RCLCPP_INFO(get_logger(),
    "DetectionCaptureNode started (camera=%s, enabled=%s)",
    camera_name_.c_str(), enabled_ ? "true" : "false");
}

void DetectionCaptureNode::detection_callback(
  const vision_msgs::msg::Detection2DArray::SharedPtr msg)
{
  if (!enabled_) return;

  bool should_save = false;
  std::string reason;

  {
    std::lock_guard<std::mutex> lock(track_mutex_);

    // Track previous confirmed set for transition detection
    std::set<int> previously_confirmed;
    for (const auto & t : tracks_) {
      if (t.state == TrackState::CONFIRMED) {
        previously_confirmed.insert(t.track_id);
      }
    }

    update_tracks(msg->detections);

    // Check for state transitions
    for (const auto & t : tracks_) {
      if (t.state == TrackState::CONFIRMED &&
          previously_confirmed.find(t.track_id) == previously_confirmed.end()) {
        should_save = true;
        reason = "object_confirmed";
        break;
      }
      if (t.state == TrackState::LOST &&
          previously_confirmed.find(t.track_id) != previously_confirmed.end()) {
        should_save = true;
        reason = "object_lost";
        break;
      }
    }

    // Prune lost tracks
    tracks_.erase(
      std::remove_if(tracks_.begin(), tracks_.end(),
        [](const TrackedObject & t) { return t.state == TrackState::LOST; }),
      tracks_.end());
  }

  // save_capture locks track_mutex_ internally, so call outside the lock
  if (should_save) {
    save_capture(reason);
  }
}

void DetectionCaptureNode::update_tracks(
  const std::vector<vision_msgs::msg::Detection2D> & detections)
{
  struct Candidate {
    size_t det_idx;
    size_t track_idx;
    double iou;
  };

  // Build candidate pairs
  std::vector<Candidate> candidates;
  for (size_t di = 0; di < detections.size(); ++di) {
    const auto & det = detections[di];
    double cx = det.bbox.center.position.x;
    double cy = det.bbox.center.position.y;
    double w = det.bbox.size_x;
    double h = det.bbox.size_y;
    double dxmin = cx - w / 2.0;
    double dymin = cy - h / 2.0;
    double dxmax = cx + w / 2.0;
    double dymax = cy + h / 2.0;

    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
      double iou = compute_iou(tracks_[ti], dxmin, dymin, dxmax, dymax);
      if (iou >= iou_threshold_) {
        candidates.push_back({di, ti, iou});
      }
    }
  }

  // Sort descending by IoU
  std::sort(candidates.begin(), candidates.end(),
    [](const Candidate & a, const Candidate & b) {
      return a.iou > b.iou;
    });

  // Greedy matching
  std::vector<bool> det_matched(detections.size(), false);
  std::vector<bool> track_matched(tracks_.size(), false);

  for (const auto & c : candidates) {
    if (det_matched[c.det_idx] || track_matched[c.track_idx]) continue;
    det_matched[c.det_idx] = true;
    track_matched[c.track_idx] = true;

    auto & track = tracks_[c.track_idx];
    const auto & det = detections[c.det_idx];

    // Update track bbox
    double cx = det.bbox.center.position.x;
    double cy = det.bbox.center.position.y;
    double w = det.bbox.size_x;
    double h = det.bbox.size_y;
    track.xmin = cx - w / 2.0;
    track.ymin = cy - h / 2.0;
    track.xmax = cx + w / 2.0;
    track.ymax = cy + h / 2.0;

    if (!det.results.empty()) {
      track.confidence = det.results[0].hypothesis.score;
      int cid = std::atoi(det.results[0].hypothesis.class_id.c_str());
      track.class_id = cid;
      if (cid >= 0 && cid < static_cast<int>(label_map_.size())) {
        track.class_name = label_map_[cid];
      }
    }

    track.frames_seen++;
    track.frames_missing = 0;

    if (track.state == TrackState::TENTATIVE &&
        track.frames_seen >= confirm_frames_) {
      track.state = TrackState::CONFIRMED;
    }
  }

  // Unmatched tracks: increment frames_missing
  for (size_t ti = 0; ti < tracks_.size(); ++ti) {
    if (!track_matched[ti]) {
      tracks_[ti].frames_missing++;
      if (tracks_[ti].frames_missing >= lost_frames_) {
        tracks_[ti].state = TrackState::LOST;
      }
    }
  }

  // Unmatched detections: create new tracks
  for (size_t di = 0; di < detections.size(); ++di) {
    if (det_matched[di]) continue;

    const auto & det = detections[di];
    TrackedObject t;
    t.track_id = next_track_id_++;

    double cx = det.bbox.center.position.x;
    double cy = det.bbox.center.position.y;
    double w = det.bbox.size_x;
    double h = det.bbox.size_y;
    t.xmin = cx - w / 2.0;
    t.ymin = cy - h / 2.0;
    t.xmax = cx + w / 2.0;
    t.ymax = cy + h / 2.0;

    t.confidence = 0.0;
    t.class_id = -1;
    if (!det.results.empty()) {
      t.confidence = det.results[0].hypothesis.score;
      t.class_id = std::atoi(det.results[0].hypothesis.class_id.c_str());
      if (t.class_id >= 0 && t.class_id < static_cast<int>(label_map_.size())) {
        t.class_name = label_map_[t.class_id];
      } else {
        t.class_name = det.results[0].hypothesis.class_id;
      }
    }

    t.frames_seen = 1;
    t.frames_missing = 0;
    t.state = (confirm_frames_ <= 1) ? TrackState::CONFIRMED : TrackState::TENTATIVE;

    tracks_.push_back(t);
  }
}

double DetectionCaptureNode::compute_iou(const TrackedObject & t,
  double xmin, double ymin, double xmax, double ymax) const
{
  double ix0 = std::max(t.xmin, xmin);
  double iy0 = std::max(t.ymin, ymin);
  double ix1 = std::min(t.xmax, xmax);
  double iy1 = std::min(t.ymax, ymax);

  double iw = std::max(0.0, ix1 - ix0);
  double ih = std::max(0.0, iy1 - iy0);
  double inter = iw * ih;

  double area_t = (t.xmax - t.xmin) * (t.ymax - t.ymin);
  double area_d = (xmax - xmin) * (ymax - ymin);
  double union_area = area_t + area_d - inter;

  if (union_area <= 0.0) return 0.0;
  return inter / union_area;
}

void DetectionCaptureNode::image_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  try {
    auto cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    std::lock_guard<std::mutex> lock(image_mutex_);
    latest_image_ = cv_ptr->image.clone();
    latest_image_header_ = msg->header;
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
      "cv_bridge exception: %s", e.what());
  }
}

void DetectionCaptureNode::save_capture(const std::string & reason)
{
  std::lock_guard<std::mutex> save_lock(save_mutex_);

  // Don't save if no session is active
  {
    std::lock_guard<std::mutex> lock(session_mutex_);
    if (!session_active_) return;
  }

  // Check min_save_interval
  auto now = this->now();
  if (last_save_time_.nanoseconds() > 0) {
    double elapsed = (now - last_save_time_).seconds();
    if (elapsed < min_save_interval_) return;
  }

  // Get the image
  cv::Mat image;
  std_msgs::msg::Header header;
  {
    std::lock_guard<std::mutex> lock(image_mutex_);
    if (latest_image_.empty()) return;
    image = latest_image_.clone();
    header = latest_image_header_;
  }

  // Check disk usage (cached, refreshed in diagnostics timer)
  if (disk_usage_bytes_ > static_cast<uint64_t>(max_disk_usage_mb_) * 1024 * 1024) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
      "Disk usage limit reached, skipping save");
    return;
  }

  // Build output path
  std::string session_dir = get_session_dir();

  // Create directory
  fs::create_directories(session_dir);

  // Build filename from timestamp
  auto stamp = header.stamp;
  std::ostringstream fname;
  fname << stamp.sec << "_" << std::setfill('0') << std::setw(9)
        << stamp.nanosec << "_" << reason;
  std::string base = fname.str();

  std::string img_path = session_dir + "/" + base + ".jpg";
  std::string json_path = session_dir + "/" + base + ".json";

  // Save JPEG
  std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality_};
  if (!cv::imwrite(img_path, image, params)) {
    RCLCPP_ERROR(get_logger(), "Failed to save image: %s", img_path.c_str());
    return;
  }

  // Build JSON sidecar
  std::ostringstream json;
  json << "{\n";
  json << "  \"timestamp_sec\": " << stamp.sec << ",\n";
  json << "  \"timestamp_nsec\": " << stamp.nanosec << ",\n";
  json << "  \"camera_name\": \"" << camera_name_ << "\",\n";
  json << "  \"trigger_reason\": \"" << reason << "\",\n";
  json << "  \"image_width\": " << image.cols << ",\n";
  json << "  \"image_height\": " << image.rows << ",\n";
  json << "  \"detections\": [";

  {
    std::lock_guard<std::mutex> lock(track_mutex_);
    bool first = true;
    for (const auto & t : tracks_) {
      if (t.state == TrackState::LOST) continue;
      if (!first) json << ",";
      first = false;
      json << "\n    {";
      json << "\"track_id\": " << t.track_id << ", ";
      json << "\"class_id\": " << t.class_id << ", ";
      json << "\"class_name\": \"" << t.class_name << "\", ";
      json << "\"confidence\": " << std::fixed << std::setprecision(4) << t.confidence << ", ";
      json << "\"bbox\": [" << static_cast<int>(t.xmin) << ", "
           << static_cast<int>(t.ymin) << ", "
           << static_cast<int>(t.xmax) << ", "
           << static_cast<int>(t.ymax) << "], ";
      json << "\"state\": \"" << (t.state == TrackState::CONFIRMED ? "confirmed" : "tentative") << "\"";
      json << "}";
    }
  }

  json << "\n  ]\n}\n";

  std::ofstream jf(json_path);
  if (jf.is_open()) {
    jf << json.str();
    jf.close();
  }

  // Update disk usage estimate (avoids full rescan)
  if (fs::exists(img_path)) {
    disk_usage_bytes_ += fs::file_size(img_path);
  }
  if (fs::exists(json_path)) {
    disk_usage_bytes_ += fs::file_size(json_path);
  }

  last_save_time_ = now;
  total_saves_++;
  {
    std::lock_guard<std::mutex> lock(session_mutex_);
    session_saves_++;
  }

  RCLCPP_DEBUG(get_logger(), "Saved capture: %s (%s)", base.c_str(), reason.c_str());
}

void DetectionCaptureNode::periodic_timer_callback()
{
  if (!enabled_) return;

  bool has_confirmed = false;
  {
    std::lock_guard<std::mutex> lock(track_mutex_);
    for (const auto & t : tracks_) {
      if (t.state == TrackState::CONFIRMED) {
        has_confirmed = true;
        break;
      }
    }
  }

  if (has_confirmed || save_negatives_) {
    save_capture("periodic");
  }
}

void DetectionCaptureNode::diagnostics_timer_callback()
{
  // Refresh disk usage cache (1 Hz is acceptable for a directory scan)
  if (fs::exists(output_dir_)) {
    disk_usage_bytes_ = compute_disk_usage(output_dir_);
  }

  int num_confirmed = 0;
  int num_tentative = 0;
  int num_tracks = 0;
  {
    std::lock_guard<std::mutex> lock(track_mutex_);
    num_tracks = static_cast<int>(tracks_.size());
    for (const auto & t : tracks_) {
      if (t.state == TrackState::CONFIRMED) num_confirmed++;
      else if (t.state == TrackState::TENTATIVE) num_tentative++;
    }
  }

  int sess_saves;
  {
    std::lock_guard<std::mutex> lock(session_mutex_);
    sess_saves = session_saves_;
  }

  uint64_t usage_mb = disk_usage_bytes_ / (1024 * 1024);

  std::ostringstream json;
  json << "{";
  json << "\"camera\": \"" << camera_name_ << "\", ";
  json << "\"enabled\": " << (enabled_ ? "true" : "false") << ", ";
  json << "\"session\": \"" << session_name_ << "\", ";
  json << "\"session_active\": " << (session_active_ ? "true" : "false") << ", ";
  json << "\"tracks\": " << num_tracks << ", ";
  json << "\"confirmed\": " << num_confirmed << ", ";
  json << "\"tentative\": " << num_tentative << ", ";
  json << "\"total_saves\": " << total_saves_ << ", ";
  json << "\"session_saves\": " << sess_saves << ", ";
  json << "\"disk_usage_mb\": " << usage_mb;
  json << "}";

  std_msgs::msg::String msg;
  msg.data = json.str();
  diag_pub_->publish(msg);
}

void DetectionCaptureNode::enable_callback(
  const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
  std::shared_ptr<std_srvs::srv::SetBool::Response> res)
{
  enabled_ = req->data;

  if (enabled_ && !img_sub_) {
    img_sub_ = create_subscription<sensor_msgs::msg::Image>(
      "rgb/image_raw", rclcpp::SensorDataQoS(),
      std::bind(&DetectionCaptureNode::image_callback, this,
        std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "Capture enabled, subscribed to image topic");
  } else if (!enabled_ && img_sub_) {
    img_sub_.reset();
    RCLCPP_INFO(get_logger(), "Capture disabled, unsubscribed from image topic");
  }

  res->success = true;
  res->message = enabled_ ? "Capture enabled" : "Capture disabled";
}

void DetectionCaptureNode::start_session_callback(
  const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
  std::shared_ptr<std_srvs::srv::Trigger::Response> res)
{
  std::lock_guard<std::mutex> lock(session_mutex_);

  // Re-read session_name parameter in case it was changed dynamically
  session_name_ = get_parameter("session_name").as_string();

  session_active_ = true;
  session_saves_ = 0;

  std::string dir = get_session_dir();
  fs::create_directories(dir);

  res->success = true;
  res->message = "Session '" + session_name_ + "' started at " + dir;
  RCLCPP_INFO(get_logger(), "Session started: %s", session_name_.c_str());
}

void DetectionCaptureNode::stop_session_callback(
  const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
  std::shared_ptr<std_srvs::srv::Trigger::Response> res)
{
  std::lock_guard<std::mutex> lock(session_mutex_);

  if (!session_active_) {
    res->success = false;
    res->message = "No active session";
    return;
  }

  session_active_ = false;
  res->success = true;
  res->message = "Session '" + session_name_ + "' stopped (" +
    std::to_string(session_saves_) + " captures)";
  RCLCPP_INFO(get_logger(), "Session stopped: %s (%d captures)",
    session_name_.c_str(), session_saves_);
}

std::string DetectionCaptureNode::get_session_dir() const
{
  return output_dir_ + "/" + session_name_ + "/" + camera_name_;
}

uint64_t DetectionCaptureNode::compute_disk_usage(const std::string & path) const
{
  uint64_t total = 0;
  try {
    for (const auto & entry : fs::recursive_directory_iterator(path,
           fs::directory_options::skip_permission_denied)) {
      if (entry.is_regular_file()) {
        total += entry.file_size();
      }
    }
  } catch (const fs::filesystem_error &) {
    // Ignore errors during iteration
  }
  return total;
}

}  // namespace oak_detection_utils

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(oak_detection_utils::DetectionCaptureNode)
