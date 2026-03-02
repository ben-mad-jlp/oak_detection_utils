#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="$(uname -m)"
DEPTHAI_INSTALL="$SCRIPT_DIR/depthai_install"

# Source ROS 2 Jazzy
source /opt/ros/jazzy/setup.bash

# --- depthai pre-built install (architecture-specific) ---
if [ ! -d "$DEPTHAI_INSTALL/lib" ]; then
    echo "ERROR: depthai_install not found at $DEPTHAI_INSTALL"
    echo ""
    echo "This directory contains pre-built depthai libraries and is architecture-specific."
    echo "Detected architecture: $ARCH"
    echo ""
    echo "Option 1: Copy a pre-built depthai_install for your architecture"
    echo "  - aarch64: copy from an existing ARM64 machine"
    echo "  - x86_64:  copy from an existing AMD64 machine"
    echo ""
    echo "Option 2: Build from source"
    echo "  git clone --recursive https://github.com/luxonis/depthai-core.git /tmp/depthai-core"
    echo "  cd /tmp/depthai-core"
    echo "  cmake -S. -Bbuild -DCMAKE_INSTALL_PREFIX=$DEPTHAI_INSTALL -DBUILD_SHARED_LIBS=ON"
    echo "  cmake --build build -j\$(nproc)"
    echo "  cmake --install build"
    echo "  touch $DEPTHAI_INSTALL/COLCON_IGNORE"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

# Verify architecture matches
INSTALLED_ARCH="$(file -b "$DEPTHAI_INSTALL/lib/"libdepthai*.so 2>/dev/null | head -1)"
case "$INSTALLED_ARCH" in
    *aarch64*) INSTALLED_FOR="aarch64" ;;
    *x86-64*)  INSTALLED_FOR="x86_64" ;;
    *)         INSTALLED_FOR="unknown" ;;
esac

if [ "$INSTALLED_FOR" != "unknown" ] && [ "$INSTALLED_FOR" != "$ARCH" ]; then
    echo "ERROR: Architecture mismatch!"
    echo "  This machine:     $ARCH"
    echo "  depthai_install:  $INSTALLED_FOR"
    echo ""
    echo "You need to rebuild depthai_install for $ARCH. See instructions above (run without depthai_install to see them)."
    exit 1
fi

# Set up depthai as underlay
# This provides depthai_v3, XLink, and other dependencies
export CMAKE_PREFIX_PATH="$DEPTHAI_INSTALL:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$DEPTHAI_INSTALL/lib:${LD_LIBRARY_PATH:-}"

# Build workspace
cd "$SCRIPT_DIR"
colcon build

# Source workspace overlay
source "$SCRIPT_DIR/install/setup.bash"

echo ""
echo "Workspace ready ($ARCH). You can now run:"
echo "  ros2 launch oak_detection_utils oak_yolo.launch.py"
