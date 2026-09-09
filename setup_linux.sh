#!/usr/bin/env bash

set -euo pipefail

if [[ $EUID -eq 0 ]]; then
    apt-get update
    apt-get install -y \
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
        nvtop
else
    sudo apt-get update
    sudo apt-get install -y \
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
        nvtop
fi

conda init bash
eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "venv"; then
    conda create -y -n venv python=3.12
fi

conda activate venv
python --version
pip install -r rent_requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu132
