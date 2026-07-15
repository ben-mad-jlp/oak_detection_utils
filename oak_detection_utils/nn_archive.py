"""Reading v3 NNArchive metadata.

The archive is the single source of truth for a model's label map and input size.
Anything that maps detections back onto an image needs the input size — detections
are reported in NN pixel space, not published-image space — so this lives here
rather than being re-derived (or worse, hardcoded) per package.
"""

import io
import json
import tarfile

# NCHW; the shape used when a v3 archive omits its input shape.
_DEFAULT_INPUT_SHAPE = [1, 3, 416, 416]


def read_nn_archive(archive_path):
    """Read label map and input size from a v3 NNArchive (.tar.xz).

    :param archive_path: Full path to the .tar.xz archive.
    :returns: (label_map, input_size) — the class-name list and the square input
        edge length in pixels.
    :raises RuntimeError: if the archive is not a v3 NNArchive.
    :raises OSError, tarfile.TarError, KeyError: if it cannot be read.
    """
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
    shape = inputs[0].get("shape", _DEFAULT_INPUT_SHAPE) if inputs else _DEFAULT_INPUT_SHAPE
    input_size = shape[2]  # NCHW

    return label_map, input_size
