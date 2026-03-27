# oak_detection_utils

ROS 2 Jazzy composable nodes that bridge OAK camera detections from `depthai_ros_driver` into
the rest of the JLP system. Runs on the **mo** computer as part of the nutting station vision
pipeline.

## Overview

`depthai_ros_driver` publishes raw `vision_msgs/Detection2DArray` detections. This package adds
three composable nodes that live in the same driver container and handle:

1. **DetectionBridgeNode** — converts raw driver detections to `depthai_yolo_msgs/BoundingBoxes`
2. **DetectionOverlayNode** — draws bounding boxes on the RGB image for monitoring
3. **DetectionCaptureNode** — saves labeled training images triggered by object events

## Nodes

### DetectionBridgeNode

Converts `vision_msgs/Detection2DArray` output from the NN pipeline into
`depthai_yolo_msgs/BoundingBoxes`. Applies sigmoid normalization when raw logits are outside
`[0, 1]` and filters by confidence threshold.

**Subscribed topics:**
| Topic | Type | Description |
|-------|------|-------------|
| `/oak/nn/detections` | `vision_msgs/Detection2DArray` | Raw NN detections from driver (remapped per camera name) |

**Published topics:**
| Topic | Type | Description |
|-------|------|-------------|
| `~/detections` | `depthai_yolo_msgs/BoundingBoxes` | Normalized, labeled bounding boxes |

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `label_map` | `string[]` | `[]` | Ordered class names matching NN output indices |
| `input_size` | `int` | `416` | NN input resolution (pixels, square) |
| `confidence_threshold` | `double` | `0.25` | Minimum confidence to pass detections |

---

### DetectionOverlayNode

Synchronizes the RGB image and detection stream, draws bounding boxes with class labels and
confidence, and publishes a visualization image at a throttled rate. Optionally shades the
dead zone (portions of the image not seen by the NN due to square cropping).

**Subscribed topics:**
| Topic | Type | Description |
|-------|------|-------------|
| `/oak/rgb/image_raw` | `sensor_msgs/Image` | Raw RGB frame (remapped per camera name) |
| `/oak/nn/detections` | `vision_msgs/Detection2DArray` | Raw NN detections (remapped per camera name) |

**Published topics:**
| Topic | Type | Description |
|-------|------|-------------|
| `~/color/overlay` | `sensor_msgs/Image` | BGR image with bounding boxes drawn |

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `label_map` | `string[]` | `[]` | Ordered class names |
| `input_size` | `int` | `416` | NN input resolution (pixels, square) |
| `publish_rate` | `double` | `5.0` | Overlay publish rate (Hz) |
| `show_dead_zone` | `bool` | `true` | Shade image regions outside NN crop |

---

### DetectionCaptureNode

Saves JPEG training images with JSON sidecar files triggered by object lifecycle events
(object confirmed, object lost) and/or on a periodic interval. Uses IoU-based multi-object
tracking to distinguish new detections from existing ones. Saves only when a session is active.

**Subscribed topics:**
| Topic | Type | Description |
|-------|------|-------------|
| `/oak/rgb/image_raw` | `sensor_msgs/Image` | RGB frame to capture (subscribed only when enabled) |
| `/oak/nn/detections` | `vision_msgs/Detection2DArray` | Detections for tracking |

**Published topics:**
| Topic | Type | Description |
|-------|------|-------------|
| `~/diagnostics` | `std_msgs/String` | JSON string with capture stats (1 Hz) |

**Services:**
| Service | Type | Description |
|---------|------|-------------|
| `~/enable` | `std_srvs/SetBool` | Enable or disable image capture |
| `~/start_session` | `std_srvs/Trigger` | Begin a named capture session |
| `~/stop_session` | `std_srvs/Trigger` | End the current session |

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `camera_name` | `string` | `""` | Camera identifier; derived from namespace if empty |
| `session_name` | `string` | `"default"` | Subdirectory name under `output_dir/` |
| `output_dir` | `string` | `"~/capture"` | Root directory for saved images |
| `enabled` | `bool` | `false` | Start capturing immediately on launch |
| `min_save_interval` | `double` | `2.0` | Minimum seconds between any two saves |
| `periodic_interval` | `double` | `30.0` | Seconds between periodic captures when objects are present |
| `confirm_frames` | `int` | `5` | Detection frames required to confirm a track |
| `lost_frames` | `int` | `15` | Missed frames before a track is marked lost |
| `iou_threshold` | `double` | `0.3` | IoU threshold for associating detections to tracks |
| `save_negatives` | `bool` | `false` | Save frames even when no objects are detected |
| `max_disk_usage_mb` | `int` | `2048` | Stop saving when output_dir exceeds this size |
| `jpeg_quality` | `int` | `85` | JPEG compression quality (0–100) |
| `label_map` | `string[]` | `[]` | Ordered class names |

**Output structure:**
```
~/capture/{session_name}/{camera_name}/{timestamp}_{reason}.jpg
~/capture/{session_name}/{camera_name}/{timestamp}_{reason}.json
```
Trigger reasons: `object_confirmed`, `object_lost`, `periodic`.

---

## Launch Files

### `oak_yolo.launch.py` — Single camera

Starts `depthai_ros_driver` and loads all three nodes into the driver's composable container.

```bash
ros2 launch oak_detection_utils oak_yolo.launch.py nn_config:=yolov8n-nut-640
ros2 launch oak_detection_utils oak_yolo.launch.py nn_config:=yolov8n-qr_code-640 name:=oak_qr mx_id:=<device_id>
```

| Argument | Default | Description |
|----------|---------|-------------|
| `name` | `oak` | Camera name; sets topic namespace and container name |
| `nn_package` | `""` | Package containing the NN config. If empty, `nn_config` is an absolute path stem. |
| `nn_config` | *(required)* | NN config path stem (no extension). Relative to `nn_package` share dir, or absolute if `nn_package` is empty. |
| `mx_id` | `""` | MyriadX device ID (empty = first available) |
| `capture_rate` | `30.0` | Seconds between periodic captures |
| `session_name` | `default` | Capture session name |

### `oak_yolo_multi.launch.py` — Multiple cameras

Launches one driver instance per camera using a hardcoded list of MyriadX device IDs. See
the launch file for the ID list and per-camera naming convention.

---

## NN Config Files

NN configs are owned by the mission packages that use them (e.g. `qr_alignment/nn/`, `nutting_station_detection/nn/`). Legacy v2 blob files live alongside their JSON configs in the same `nn/` directory.

Two formats are supported:

**v3 NNArchive** (preferred): JSON with `config_version: "1.0"` paired with a `.tar.xz` archive.
The launch file reads label map and input size from the JSON and passes the archive path to the
driver.

**Legacy v2**: JSON with `mappings.labels` and `model.model_name`. Relative blob paths are
resolved from `config/nn/blobs/`.

### Available configs

See `qr_alignment/nn/` for QR configs and `nutting_station_detection/nn/` for nut/stud configs.

---

## Setup and Build

Clone all dependencies into the workspace using `ros_ws.repos`:

```bash
vcs import ~/Projects/code/ros_ws/src < ~/Projects/code/ros_ws/src/oak_detection_utils/ros.repos
rosdep install --from-paths ~/Projects/code/ros_ws/src --ignore-src -y
cd ~/Projects/code/ros_ws && colcon build
```

| Repo | Branch | Purpose |
|------|--------|---------|
| `ben-mad-jlp/depthai-ros` | `kilted-local-nnarchive` | depthai ROS 2 driver (forked, NNArchive support) |
| `ben-mad-jlp/depthai_yolo_msgs` | `master` | BoundingBox message definitions |
| `thebyohazard/depthai_vendor` | `master` | depthai-core SDK vendor package (builds from source) |

`depthai_vendor` builds depthai-core from source and takes ~45 minutes on first build. Once built,
add an untracked `COLCON_IGNORE` to `src/depthai_vendor/` to skip it in subsequent builds.
