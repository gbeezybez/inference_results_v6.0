#!/bin/bash

apt-get update
apt-get install git-lfs
git lfs install
cd /model
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
