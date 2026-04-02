import io
import json
import os
import tarfile
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode


def launch_setup(context, *args, **kwargs):
    driver_dir = get_package_share_directory("depthai_ros_driver")

    nn_package = LaunchConfiguration("nn_package").perform(context)
    nn_config = LaunchConfiguration("nn_config").perform(context)
    mx_id = LaunchConfiguration("mx_id").perform(context)
    capture_rate = float(LaunchConfiguration("capture_rate").perform(context))
    session_name = LaunchConfiguration("session_name").perform(context)
    namespace = LaunchConfiguration("namespace").perform(context)

    if nn_package:
        base_dir = get_package_share_directory(nn_package)
        archive_path = os.path.join(base_dir, nn_config + ".tar.xz")
    else:
        archive_path = nn_config + ".tar.xz"

    # Read label_map and input_size from config.json inside the NNArchive (.tar.xz)
    with tarfile.open(archive_path, "r:xz") as tar:
        config_member = tar.getmember("config.json")
        with tar.extractfile(config_member) as f:
            nn_json = json.load(io.TextIOWrapper(f))

    if "config_version" not in nn_json:
        raise RuntimeError(
            f"{archive_path} does not contain a v3 NNArchive config.json "
            "(missing 'config_version' key)"
        )

    model = nn_json.get("model", {})
    heads = model.get("heads", [{}])
    label_map = heads[0].get("metadata", {}).get("classes", []) if heads else []
    inputs = model.get("inputs", [{}])
    shape = inputs[0].get("shape", [1, 3, 416, 416]) if inputs else [1, 3, 416, 416]
    input_size = shape[2]  # NCHW format
    nn_model_path = archive_path

    # Generate camera params YAML with runtime values (v3 driver format)
    camera_params = {
        "/**": {
            "ros__parameters": {
                "driver": {
                    "i_enable_ir": False,
                },
                "pipeline_gen": {
                    "i_pipeline_type": "RGB",
                    "i_nn_type": "rgb",
                },
                "nn": {
                    "i_nn_model": nn_model_path,
                },
                "rgb": {
                    "i_publish_topic": True,
                    "i_fps": 30.0,
                    "i_width": input_size,
                    "i_height": input_size,
                },
            }
        }
    }
    if mx_id:
        camera_params["/**"]["ros__parameters"]["driver"]["i_device_id"] = mx_id

    # Write to a temp file so the driver can load it
    params_fd, params_path = tempfile.mkstemp(suffix=".yaml", prefix="oak_params_")
    with os.fdopen(params_fd, "w") as f:
        yaml.dump(camera_params, f, default_flow_style=False)

    camera_name = LaunchConfiguration("name").perform(context)

    # Include the official depthai_ros_driver v3 launch
    driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(driver_dir, "launch", "driver.launch.py")
        ),
        launch_arguments={
            "name": camera_name,
            "namespace": namespace,
            "params_file": params_path,
            "camera_model": "OAK-1",
            "pointcloud.enable": "false",
            "use_rviz": "false",
        }.items(),
    )

    # Load bridge and overlay nodes into the driver's container.
    #
    # The container is created by driver.launch.py at /{namespace}/{camera_name}_container.
    # We must use the absolute path here because there is no PushRosNamespace context
    # to resolve a relative name — a relative name would look for /{camera_name}_container
    # at the global root and fail to find the container.
    container_name = f"{camera_name}_container"
    abs_container = f"/{namespace}/{container_name}" if namespace else f"/{container_name}"
    load_nodes = LoadComposableNodes(
        target_container=abs_container,
        composable_node_descriptions=[
            ComposableNode(
                package="oak_detection_utils",
                plugin="oak_detection_utils::DetectionBridgeNode",
                name=f"{camera_name}_bridge",
                namespace=namespace,
                parameters=[{
                    "label_map": label_map,
                    "input_size": input_size,
                }],
                remappings=[
                    ("nn/detections", f"{camera_name}/nn/detections"),
                ],
            ),
            ComposableNode(
                package="oak_detection_utils",
                plugin="oak_detection_utils::DetectionOverlayNode",
                name=f"{camera_name}_overlay",
                namespace=namespace,
                parameters=[{
                    "label_map": label_map,
                    "input_size": input_size,
                    "publish_rate": 5.0,
                    "show_dead_zone": True,
                }],
                remappings=[
                    ("rgb/image_raw", f"{camera_name}/rgb/image_raw"),
                    ("nn/detections", f"{camera_name}/nn/detections"),
                ],
            ),
            ComposableNode(
                package="oak_detection_utils",
                plugin="oak_detection_utils::DetectionCaptureNode",
                name=f"{camera_name}_capture",
                namespace=namespace,
                parameters=[{
                    "label_map": label_map,
                    "camera_name": camera_name,
                    "session_name": session_name,
                    "enabled": True,
                    "min_save_interval": capture_rate,
                    "periodic_interval": capture_rate,
                }],
                remappings=[
                    ("rgb/image_raw", f"{camera_name}/rgb/image_raw"),
                    ("nn/detections", f"{camera_name}/nn/detections"),
                ],
            ),
        ],
    )

    return [driver_launch, load_nodes]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "name",
            default_value="oak",
            description="Camera node name (alphanumeric + underscores only)",
        ),
        DeclareLaunchArgument(
            "nn_package",
            default_value="",
            description="Package containing the NN config files. If empty, nn_config is treated as an absolute path stem.",
        ),
        DeclareLaunchArgument(
            "nn_config",
            description="NN config path stem (no extension). Relative to nn_package share dir if nn_package is set, otherwise absolute.",
        ),
        DeclareLaunchArgument(
            "mx_id",
            default_value="",
            description="MyriadX device ID (empty for first available)",
        ),
        DeclareLaunchArgument(
            "capture_rate",
            default_value="30.0",
            description="Minimum interval (seconds) between image captures",
        ),
        DeclareLaunchArgument(
            "namespace",
            default_value="",
            description="ROS namespace for the driver and composable nodes. "
                        "Pass explicitly to prevent outer launch context from leaking in.",
        ),
        DeclareLaunchArgument(
            "session_name",
            default_value="default",
            description="Capture session name (images saved to ~/capture/{session_name}/{camera}/)",
        ),
        OpaqueFunction(function=launch_setup),
    ])
