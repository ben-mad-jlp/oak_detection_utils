import json
import os
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
    pkg_dir = get_package_share_directory("oak_detection_utils")
    driver_dir = get_package_share_directory("depthai_ros_driver")

    nn_config_name = LaunchConfiguration("nn_config").perform(context)
    mx_id = LaunchConfiguration("mx_id").perform(context)

    nn_config_path = os.path.join(pkg_dir, "config", "nn", f"{nn_config_name}.json")

    # Read label_map and input_size from the NN JSON config (v3 NNArchive format)
    with open(nn_config_path) as f:
        nn_json = json.load(f)

    # Support both v3 NNArchive config and legacy v2 config formats
    if "config_version" in nn_json:
        # v3 NNArchive config.json format
        model = nn_json.get("model", {})
        heads = model.get("heads", [{}])
        label_map = heads[0].get("metadata", {}).get("classes", []) if heads else []
        inputs = model.get("inputs", [{}])
        shape = inputs[0].get("shape", [1, 3, 416, 416]) if inputs else [1, 3, 416, 416]
        input_size = shape[2]  # NCHW format
        # For v3, use the .tar.xz NNArchive (same base name as the JSON config)
        archive_path = os.path.join(
            pkg_dir, "config", "nn", f"{nn_config_name}.tar.xz"
        )
        nn_model_path = archive_path
    else:
        # Legacy v2 config format
        label_map = nn_json.get("mappings", {}).get("labels", [])
        nn_model_path = nn_json.get("model", {}).get("model_name", "")
        input_size = 416
        input_size_str = nn_json.get("nn_config", {}).get("input_size", "416x416")
        if "x" in input_size_str:
            input_size = int(input_size_str.split("x")[0])

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

    camera_name = "oak"

    # Include the official depthai_ros_driver v3 launch
    driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(driver_dir, "launch", "driver.launch.py")
        ),
        launch_arguments={
            "name": camera_name,
            "params_file": params_path,
            "camera_model": "OAK-D-LITE",
            "pointcloud.enable": "false",
            "use_rviz": "false",
        }.items(),
    )

    # Load bridge and overlay nodes into the driver's container
    container_name = f"{camera_name}_container"
    load_nodes = LoadComposableNodes(
        target_container=container_name,
        composable_node_descriptions=[
            ComposableNode(
                package="oak_detection_utils",
                plugin="oak_detection_utils::DetectionBridgeNode",
                name="oak_d",
                parameters=[{
                    "label_map": label_map,
                    "input_size": input_size,
                }],
            ),
            ComposableNode(
                package="oak_detection_utils",
                plugin="oak_detection_utils::DetectionOverlayNode",
                name="detection_overlay",
                namespace="oak_d",
                parameters=[{
                    "label_map": label_map,
                    "input_size": input_size,
                    "publish_rate": 5.0,
                    "show_dead_zone": True,
                }],
            ),
        ],
    )

    return [driver_launch, load_nodes]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "nn_config",
            default_value="yolov8n-qr_code-640",
            description="NN config name (matches JSON filename in config/nn/; "
            "v3 configs expect a matching .tar.xz NNArchive)",
        ),
        DeclareLaunchArgument(
            "mx_id",
            default_value="",
            description="MyriadX device ID (empty for first available)",
        ),
        OpaqueFunction(function=launch_setup),
    ])
