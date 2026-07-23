#!/usr/bin/env bash
# Pull the latest published image and (re)create the Toolbx/Distrobox container.
#   ./toolbox/refresh-toolbox.sh
set -e

# Built from .devops/strix-halo.Dockerfile and pushed manually (lowercase owner/repo).
IMAGE="docker.io/higaetan/strix-halo-llamacpp-toolbox:rocm-7.14"
NAME="llama-rocm-strixhalo"
OPTS="--device /dev/dri --device /dev/kfd --group-add video --group-add render --group-add sudo --security-opt seccomp=unconfined"

# toolbox on Fedora, distrobox on Debian/Ubuntu.
TOOLBOX_CMD="toolbox"
if [ -f /etc/os-release ]; then
  . /etc/os-release
  if [ "$ID" = "ubuntu" ] || [ "$ID" = "debian" ]; then TOOLBOX_CMD="distrobox"; fi
fi
command -v podman >/dev/null || { echo "podman not installed"; exit 1; }
command -v "$TOOLBOX_CMD" >/dev/null || { echo "$TOOLBOX_CMD not installed"; exit 1; }

echo "🔄 Refreshing $NAME ($IMAGE)"
$TOOLBOX_CMD list | grep -q "$NAME" && $TOOLBOX_CMD rm -f "$NAME"
podman pull "$IMAGE"
$TOOLBOX_CMD create "$NAME" --image "$IMAGE" -- $OPTS
echo "✅ ready — enter with: $TOOLBOX_CMD enter $NAME"
