#!/bin/bash

# login to huggingface
# hf login sometimes causes race condition. Executing before everything else helps.
pip install -U huggingface_hub[cli]
huggingface-cli login --token YOUR_HF_TOKEN

# install loadgen whl (not sure why this is not persisted in sqsh)
pip install --user --force-reinstall build/inference/loadgen/mlcommons_loadgen*.whl

# create links
make link_dirs

