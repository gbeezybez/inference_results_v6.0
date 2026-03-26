#!/bin/bash
set -e

SETUP_DIR=$(dirname -- $0)
ARCH=${ARCH:-$(rocminfo | grep "Name:" | grep "gfx" | awk 'NR==1' | awk '{print $2}')}
VLLM_IMAGE_NAME=rocm/vllm-dev:nightly_0610_rc2_0610_rc2_20250605
PATCHES="--apply-vllm-patches --apply-llama2-patches"
if [[ "$ARCH" == "gfx950" ]]; then
    VLLM_IMAGE_NAME=rocm/7.0-preview:rocm7.0_preview_ubuntu_22.04_vllm_0.8.5_mi35X_prealpha1
    PATCHES="$PATCHES --apply-aiter-patches"
elif [[ "$ARCH" == "gfx942" ]]; then
    PATCHES="$PATCHES --custom-fa-branch b7d29fb"
fi
# Create llama2-70b docker
bash $SETUP_DIR/build_docker.sh --base-docker-image $VLLM_IMAGE_NAME --arch $ARCH $PATCHES
image_name=$(cat mlperf_image_name.txt)
if [ ! -z "$1" ]; then docker tag $image_name $1; fi
