#!/usr/bin/env bash
# Launch Momentum with NVIDIA PRIME render offload.
# Required on hybrid Intel+NVIDIA setups where NVIDIA owns GLX
# but Intel modesetting owns the display outputs.
cd "$(dirname "$0")"
exec env \
  __NV_PRIME_RENDER_OFFLOAD=1 \
  __GLX_VENDOR_LIBRARY_NAME=nvidia \
  ./target/release/momentum "$@"
