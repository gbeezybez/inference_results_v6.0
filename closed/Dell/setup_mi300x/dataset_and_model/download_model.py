# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse, os, sys
import subprocess
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="/model", type=str, help="Path to the model location")
args = parser.parse_args()
token_hf=os.getenv("HUGGINGFACE_ACCESS_TOKEN", "")

models_hf = (
    ("meta-llama/Llama-2-70b-chat-hf", "llama2-70b-chat-hf/orig"),
    # Premade models (requires token)
    # ("amd/Llama-2-70b-chat-hf_FP8_MLPerf_V2", "llama2-70b-chat-hf/fp8_quantized"),
    # ("amd/Mixtral-8x7B-Instruct-v0.1_FP8_MLPerf_V2", "mixtral-8x7b/fp8_quantized"),
)

models_rclone = (
    ("mixtral_8x7b/mixtral-8x7b-instruct-v0.1", "mixtral-8x7b/orig"),
)

if token_hf == "" and models_hf:
    print("Missing HUGGINGFACE_ACCESS_TOKEN")
    for model_id, model_path in models_hf:
        print(f"Skipping {model_id}")
else:
    for model_id, model_path in models_hf:
        # Create the directory
        download_dir = f"{args.model_dir}/{model_path}"
        os.makedirs(download_dir,  exist_ok=True)

        snapshot_download(
            repo_id=model_id,
            local_dir=download_dir,
            max_workers=32,
            token=token_hf,
        )
        print(f"Succesfully downloaded {model_id}")

if models_rclone:
    rclone_config = "rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com"
    rclone_copy = "rclone copy mlc-inference:mlcommons-inference-wg-public/"

    subprocess.run(rclone_config, shell=True)

    for model_id, model_path in models_rclone:
        subprocess.run(rclone_copy + model_id + " /model/" + model_path + " -P", shell=True)
        print(f"Succesfully downloaded {model_id}")
