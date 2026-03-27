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


CAMERA_IDS = [
    "19443010F18CDB2C00",
    "19443010A193172F00",
    "14442C10513E1CD000",
    "14442C10C13F1ED000",
    "1944301001CB152F00",
]


def launch_setup(context, *args, **kwargs):
    driver_dir = get_package_share_directory("depthai_ros_driver")

    nn_package = LaunchConfiguration("nn_package").perform(context)
    nn_config = LaunchConfiguration("nn_config").perform(context)

    if nn_package:
        base_dir = get_package_share_directory(nn_package)
        nn_config_path = os.path.join(base_dir, nn_config + ".json")
        archive_path = os.path.join(base_dir, nn_config + ".tar.xz")
    else:
        nn_config_path = nn_config + ".json"
        archive_path = os.path.splitext(nn_config)[0] + ".tar.xz"

    with open(nn_config_path) as f:
        nn_json = json.load(f)

    if "config_version" in nn_json:
        model = nn_json.get("model", {})
        heads = model.get("heads", [{}])
        label_map = heads[0].get("metadata", {}).get("classes", []) if heads else []
        inputs = model.get("inputs", [{}])
        shape = inputs[0].get("shape", [1, 3, 416, 416]) if inputs else [1, 3, 416, 416]
        input_size = shape[2]
        nn_model_path = archive_path
    else:
        label_map = nn_json.get("mappings", {}).get("labels", [])
        nn_model_path = nn_json.get("model", {}).get("model_name", "")
        input_size = 416
        input_size_str = nn_json.get("nn_config", {}).get("input_size", "416x416")
        if "x" in input_size_str:
            input_size = int(input_size_str.split("x")[0])

    actions = []

    for idx, mx_id in enumerate(CAMERA_IDS):
        camera_name = f"oak_{idx}"

        camera_params = {
            "/**": {
                "ros__parameters": {
                    "driver": {
                        "i_enable_ir": False,
                        "i_device_id": mx_id,
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

        params_fd, params_path = tempfile.mkstemp(
            suffix=".yaml", prefix=f"oak_{idx}_params_"
        )
        with os.fdopen(params_fd, "w") as f:
            yaml.dump(camera_params, f, default_flow_style=False)

        driver_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(driver_dir, "launch", "driver.launch.py")
            ),
            launch_arguments={
                "name": camera_name,
                "params_file": params_path,
                "camera_model": "OAK-1-LITE",
                "pointcloud.enable": "false",
                "use_rviz": "false",
            }.items(),
        )

        container_name = f"{camera_name}_container"
        load_nodes = LoadComposableNodes(
            target_container=container_name,
            composable_node_descriptions=[
                ComposableNode(
                    package="oak_detection_utils",
                    plugin="oak_detection_utils::DetectionBridgeNode",
                    name=f"{camera_name}_bridge",
                    parameters=[{
                        "label_map": label_map,
                        "input_size": input_size,
                    }],
                    remappings=[
                        ("/oak/nn/detections", f"/{camera_name}/nn/detections"),
                    ],
                ),
                ComposableNode(
                    package="oak_detection_utils",
                    plugin="oak_detection_utils::DetectionOverlayNode",
                    name="detection_overlay",
                    namespace=f"{camera_name}_bridge",
                    parameters=[{
                        "label_map": label_map,
                        "input_size": input_size,
                        "publish_rate": 5.0,
                        "show_dead_zone": True,
                    }],
                    remappings=[
                        ("/oak/rgb/image_raw", f"/{camera_name}/rgb/image_raw"),
                        ("/oak/nn/detections", f"/{camera_name}/nn/detections"),
                    ],
                ),
                ComposableNode(
                    package="oak_detection_utils",
                    plugin="oak_detection_utils::DetectionCaptureNode",
                    name="capture",
                    namespace=f"{camera_name}_bridge",
                    parameters=[{
                        "label_map": label_map,
                        "camera_name": camera_name,
                    }],
                    remappings=[
                        ("/oak/rgb/image_raw", f"/{camera_name}/rgb/image_raw"),
                        ("/oak/nn/detections", f"/{camera_name}/nn/detections"),
                    ],
                ),
            ],
        )

        actions.append(driver_launch)
        actions.append(load_nodes)

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "nn_package",
            default_value="",
            description="Package containing the NN config files. If empty, nn_config is treated as an absolute path stem.",
        ),
        DeclareLaunchArgument(
            "nn_config",
            description="NN config path stem (no extension). Relative to nn_package share dir if nn_package is set, otherwise absolute.",
        ),
        OpaqueFunction(function=launch_setup),
    ])
