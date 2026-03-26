#!/bin/bash
# install_prerequisites.sh - Install system dependencies for MLPerf Inference
set -euo pipefail

echo "[ Info ] Installing system prerequisites..."
sudo apt update || { echo "[ Error ] Failed to update package lists."; exit 1; }

sudo apt install -y --no-install-recommends \
    git \
    build-essential \
    software-properties-common \
    ca-certificates \
    wget \
    curl \
    htop \
    zip \
    unzip \
    python3-venv \
    || { echo "[ Error ] Failed to install packages."; exit 1; }

echo "[ Done ] System prerequisites installed successfully."
