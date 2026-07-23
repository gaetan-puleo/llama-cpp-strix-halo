#!/usr/bin/env bash
# Pull the latest published image and (re)create the container.
# Works with both toolbox (Fedora) and distrobox (Ubuntu/Debian/others).
#   ./toolbox/refresh-toolbox.sh              # auto-detect
#   RUNNER=distrobox ./toolbox/refresh-toolbox.sh   # force distrobox
#   RUNNER=toolbox   ./toolbox/refresh-toolbox.sh   # force toolbox
set -e

IMAGE="docker.io/higaetan/strix-halo-llamacpp-toolbox:rocm-7.14"
NAME="strix-halo"
OPTS="--device /dev/dri --device /dev/kfd --group-add video --group-add render --group-add sudo --security-opt seccomp=unconfined"

# Pick the runner: explicit RUNNER wins; else toolbox on Fedora, distrobox elsewhere;
# else whichever is installed.
RUNNER="${RUNNER:-}"
if [ -z "$RUNNER" ]; then
  RUNNER="toolbox"
  if [ -f /etc/os-release ]; then
    . /etc/os-release
    case "$ID $ID_LIKE" in *ubuntu*|*debian*) RUNNER="distrobox";; esac
  fi
  command -v "$RUNNER" >/dev/null 2>&1 || { command -v distrobox >/dev/null 2>&1 && RUNNER=distrobox; }
  command -v "$RUNNER" >/dev/null 2>&1 || { command -v toolbox   >/dev/null 2>&1 && RUNNER=toolbox; }
fi

command -v podman   >/dev/null 2>&1 || { echo "podman not installed"; exit 1; }
command -v "$RUNNER" >/dev/null 2>&1 || { echo "$RUNNER not installed"; exit 1; }

echo "Refreshing '$NAME' via $RUNNER ($IMAGE)"
podman pull "$IMAGE"

if [ "$RUNNER" = "distrobox" ]; then
  distrobox rm -f "$NAME" 2>/dev/null || true
  distrobox create --name "$NAME" --image "$IMAGE" --additional-flags "$OPTS"
else
  toolbox list | grep -q "$NAME" && toolbox rm -f "$NAME" || true
  toolbox create "$NAME" --image "$IMAGE" -- $OPTS
fi

echo "Ready. Enter with: $RUNNER enter $NAME"
