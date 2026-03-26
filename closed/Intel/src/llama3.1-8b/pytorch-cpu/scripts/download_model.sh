#!/bin/bash
MODEL_DIR=${MODEL_DIR:-/model/}
mlcr get,ml-model,llama3,_meta-llama/Llama-3.1-8B-Instruct,_hf --outdirname=${MODEL_DIR} --hf_token=${HF_TOKEN} -j

mv ${MODEL_DIR}/repo ${MODEL_DIR}/Llama-3.1-8B-Instruct
