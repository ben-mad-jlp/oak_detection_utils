import os
import tempfile

import launch.logging
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LoadComposableNodes
from launch_ros.descriptions import ComposableNode

from oak_detection_utils.nn_archive import read_nn_archive


def _deep_merge(base, override):
    """Recursively merge override into base dict in-place."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _report_nn_framing(camera_name, input_size, rgb_width, rgb_height):
    """Surface how the NN frames the scene relative to the published stream.

    The NN samples the sensor itself, aspect-preserving, so it sees only the centered
    square of the field of view. Whenever the published stream is not square, the two
    disagree — and detections are reported in NN pixel space, not published-image
    space. That mismatch is invisible until someone maps a box onto the image, so say
    it out loud at launch.
    """
    logger = launch.logging.get_logger("oak_yolo")

    if min(rgb_width, rgb_height) < input_size:
        logger.warning(
            f"{camera_name}: published rgb is {rgb_width}x{rgb_height}, smaller than the "
            f"NN input {input_size}x{input_size}. The NN samples the sensor directly so its "
            f"accuracy is unaffected, but the published image is coarser than what the NN "
            f"sees — anything doing sub-pixel work on it is needlessly handicapped."
        )

    if rgb_width != rgb_height:
        nn_region = min(rgb_width, rgb_height)
        dead_zone = (max(rgb_width, rgb_height) - nn_region) // 2
        logger.info(
            f"{camera_name}: NN sees only the centered {nn_region}x{nn_region} square of the "
            f"{rgb_width}x{rgb_height} frame ({dead_zone} px dead zone per side). Detections "
            f"are in {input_size}x{input_size} NN space — map them onto the image through "
            f"that center crop (scale by {nn_region}/{input_size}, then offset), never by "
            f"stretching each axis independently."
        )


def launch_setup(context, *args, **kwargs):
    driver_dir = get_package_share_directory("depthai_ros_driver")

    nn_package = LaunchConfiguration("nn_package").perform(context)
    nn_config = LaunchConfiguration("nn_config").perform(context)
    mx_id = LaunchConfiguration("mx_id").perform(context)
    params_file = LaunchConfiguration("params_file").perform(context)
    capture_rate = float(LaunchConfiguration("capture_rate").perform(context))
    session_name = LaunchConfiguration("session_name").perform(context)
    namespace = LaunchConfiguration("namespace").perform(context)
    camera_name = LaunchConfiguration("name").perform(context)
    camera_model = LaunchConfiguration("camera_model").perform(context)
    rgb_width = int(LaunchConfiguration("rgb_width").perform(context))
    rgb_height = int(LaunchConfiguration("rgb_height").perform(context))

    # Per-camera "mount" anchor frame: the rsp publishes a static identity
    # transform from <name>_mount → <name> (camera body). To localize a camera,
    # publish <scene_frame> → <name>_mount dynamically (e.g. via
    # dynamic_tf_publisher driven by an apriltag-localize service). To pin a
    # camera at a known static pose instead, override parent_frame and the
    # cam_pos_*/cam_* args here at launch.
    # NOTE: cannot use LaunchConfiguration(..., default=...) here — the
    # DeclareLaunchArgument below sets default_value="" and that wins over
    # the kwarg, so the kwarg never fires. Resolve the {name}_mount default
    # explicitly when the launch arg is empty.
    parent_frame = (
        LaunchConfiguration("parent_frame").perform(context)
        or f"{camera_name}_mount"
    )
    cam_pos_x = LaunchConfiguration("cam_pos_x").perform(context)
    cam_pos_y = LaunchConfiguration("cam_pos_y").perform(context)
    cam_pos_z = LaunchConfiguration("cam_pos_z").perform(context)
    cam_roll = LaunchConfiguration("cam_roll").perform(context)
    cam_pitch = LaunchConfiguration("cam_pitch").perform(context)
    cam_yaw = LaunchConfiguration("cam_yaw").perform(context)

    # Base RGB pipeline params. rgb_width/rgb_height size the PUBLISHED stream, and
    # nothing else — the NN does not consume it. In depthai v3 the neural network
    # requests its own branch straight off the sensor at the model's input size, so
    # published resolution and NN input are independent and are tuned separately.
    # (v2's "preview" — where the published stream really was the NN input — is gone.)
    camera_params = {
        "/**": {
            "ros__parameters": {
                "driver": {
                    "i_enable_ir": False,
                },
                "pipeline_gen": {
                    "i_pipeline_type": "RGB",
                },
                "rgb": {
                    "i_publish_topic": True,
                    "i_fps": 30.0,
                    "i_width": rgb_width,
                    "i_height": rgb_height,
                },
            }
        }
    }

    label_map = []
    input_size = None
    if nn_config:
        if nn_package:
            base_dir = get_package_share_directory(nn_package)
            archive_path = os.path.join(base_dir, nn_config + ".tar.xz")
        else:
            archive_path = nn_config + ".tar.xz"

        label_map, input_size = read_nn_archive(archive_path)

        rp = camera_params["/**"]["ros__parameters"]
        rp["pipeline_gen"]["i_nn_type"] = "rgb"
        rp["nn"] = {"i_nn_model": archive_path}

        _report_nn_framing(camera_name, input_size, rgb_width, rgb_height)
    # Merge driver params from the instance params_file (e.g. i_device_id).
    # Reads the section keyed by the full node path: {namespace}/{camera_name}.
    # Accepts the key with or without a leading slash so files can follow either
    # convention (vision_detection.yaml omits it; ns1/params.yaml keeps it).
    if params_file:
        try:
            with open(params_file) as f:
                extra = yaml.safe_load(f) or {}
            node_key = f"{namespace}/{camera_name}" if namespace else camera_name
            section = extra.get(node_key) or extra.get(f"/{node_key}") or {}
            node_ros_params = section.get("ros__parameters", {})
            _deep_merge(camera_params["/**"]["ros__parameters"], node_ros_params)
        except (OSError, yaml.YAMLError):
            pass

    # mx_id launch arg overrides everything if explicitly set
    if mx_id:
        camera_params["/**"]["ros__parameters"]["driver"]["i_device_id"] = mx_id

    # Write to a temp file so the driver can load it
    params_fd, params_path = tempfile.mkstemp(suffix=".yaml", prefix="oak_params_")
    with os.fdopen(params_fd, "w") as f:
        yaml.dump(camera_params, f, default_flow_style=False)

    # Include the official depthai_ros_driver v3 launch
    driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(driver_dir, "launch", "driver.launch.py")
        ),
        launch_arguments={
            "name": camera_name,
            "namespace": namespace,
            "params_file": params_path,
            "camera_model": camera_model,
            "pointcloud.enable": "false",
            "use_rviz": "false",
            "parent_frame": parent_frame,
            "cam_pos_x": cam_pos_x,
            "cam_pos_y": cam_pos_y,
            "cam_pos_z": cam_pos_z,
            "cam_roll": cam_roll,
            "cam_pitch": cam_pitch,
            "cam_yaw": cam_yaw,
        }.items(),
    )

    inner_actions = [driver_launch]

    # Load bridge and overlay nodes into the driver's container.
    # These are all NN-dependent (they consume nn/detections and the model's
    # label_map, and input_size comes from the NN archive), so they are only
    # built and loaded when an NN is configured. With no NN, input_size is None
    # and constructing these ComposableNodes would fail parameter validation —
    # so the camera-only path skips them entirely.
    #
    # The container is created by driver.launch.py at /{namespace}/{camera_name}_container.
    # We must use the absolute path here because there is no PushRosNamespace context
    # to resolve a relative name — a relative name would look for /{camera_name}_container
    # at the global root and fail to find the container.
    if nn_config:
        container_name = f"{camera_name}_container"
        abs_container = f"/{namespace}/{container_name}" if namespace else f"/{container_name}"
        inner_actions.append(LoadComposableNodes(
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
        ))

    # Wrap in a scoped GroupAction so the inner driver.launch.py's
    # launch_arguments (parent_frame, cam_pos_*, cam_*, etc.) don't leak into
    # the parent launch context. Without this, when oak_yolo is included twice
    # (e.g. for two cameras), the second invocation reads launch arg values
    # left behind by the first invocation's inner driver.launch.py call,
    # silently cross-contaminating per-camera defaults.
    return [GroupAction(inner_actions, scoped=True)]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "name",
            default_value="oak",
            description="Camera node name (alphanumeric + underscores only). "
                        "MUST be unique across all cameras in the ROS graph "
                        "even if they live in different namespaces — see warning below.",
        ),
        # ──────────────────────────────────────────────────────────────────────
        # DO NOT try to give two cameras the same `name` (e.g. "cam") under
        # different namespaces hoping for a cleaner topology. The depthai
        # stack uses the camera node name as the prefix for all sub-sensor
        # tf frames (cam_rgb_camera_frame, cam_imu_frame, cam_*_optical_frame)
        # via depthai_bridge::TFPublisher. Two cameras with the same node
        # name will publish colliding /tf_static frames and one will
        # silently overwrite the other. We tried adding a `tf_prefix` knob
        # to decouple them; it doesn't work because the C++ side ignores it
        # for sub-sensor frames. Just give each camera a unique name.
        # ──────────────────────────────────────────────────────────────────────
        DeclareLaunchArgument(
            "nn_package",
            default_value="",
            description="Package containing the NN config files. If empty, nn_config is treated as an absolute path stem.",
        ),
        DeclareLaunchArgument(
            "nn_config",
            default_value="",
            description="NN config path stem (no extension). Relative to nn_package share dir if nn_package is set, otherwise absolute. "
                        "Leave empty to run the OAK as a plain RGB camera with no neural network (no bridge/overlay/capture nodes).",
        ),
        DeclareLaunchArgument(
            "camera_model",
            default_value="OAK-1",
            description="DepthAI camera model passed to the driver (e.g. OAK-1, OAK-D-PRO).",
        ),
        DeclareLaunchArgument(
            "rgb_width",
            default_value="1280",
            description="Published RGB stream width (pixels). Independent of the NN, which "
                        "samples the sensor on its own branch at the model's input size. "
                        "A non-square stream means the NN sees only its centered square.",
        ),
        DeclareLaunchArgument(
            "rgb_height",
            default_value="720",
            description="Published RGB stream height (pixels). See rgb_width.",
        ),
        DeclareLaunchArgument(
            "mx_id",
            default_value="",
            description="MyriadX device ID. Overrides params_file if set.",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value="",
            description="Instance params YAML. Driver params are read from the "
                        "{namespace}/{name} section. mx_id arg takes precedence if set.",
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
        # ── Camera mount / pose ────────────────────────────────────────────────
        # By default we anchor the camera body to a per-camera "mount" frame
        # ({name}_mount) at identity, so a separate publisher (e.g. a static
        # transform from a calibration node, or dynamic_tf_publisher updated
        # by an apriltag-localize service) can own the {scene} → {name}_mount
        # edge. Override these args if you'd rather pin the camera at a
        # known static pose at launch time.
        DeclareLaunchArgument(
            "parent_frame",
            default_value="",
            description="Parent frame the camera body attaches to. "
                        "Defaults to '{name}_mount' if empty — left for an external "
                        "publisher (e.g. dynamic_tf_publisher) to position dynamically.",
        ),
        DeclareLaunchArgument("cam_pos_x", default_value="0.0",
            description="Camera body X offset from parent_frame (meters)."),
        DeclareLaunchArgument("cam_pos_y", default_value="0.0",
            description="Camera body Y offset from parent_frame (meters)."),
        DeclareLaunchArgument("cam_pos_z", default_value="0.0",
            description="Camera body Z offset from parent_frame (meters)."),
        DeclareLaunchArgument("cam_roll", default_value="0.0",
            description="Camera body roll w.r.t. parent_frame (radians)."),
        DeclareLaunchArgument("cam_pitch", default_value="0.0",
            description="Camera body pitch w.r.t. parent_frame (radians)."),
        DeclareLaunchArgument("cam_yaw", default_value="0.0",
            description="Camera body yaw w.r.t. parent_frame (radians)."),
        OpaqueFunction(function=launch_setup),
    ])
