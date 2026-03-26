#!/usr/bin/env python3
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


__doc__ = """
    Scripts that calculates FID between TRT and Torch generated images and CLIP from TRT generated images
    The L0 accuracy tester uses torchmetrics FID and CLIP methods which are different from the official accuracy script
"""

import argparse
import numpy as np
import pandas as pd
import torch

from PIL import Image
from pathlib import Path
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance

from code.common import logging

class CoCoAccuracyTester:
    """
    Post processing to calculate FID and CLIP
    ref: https://github.com/mlcommons/inference/blob/master/text_to_image/tools/accuracy_coco.py
    """

    def __init__(self, raw_captions, trt_image_dir, pytorch_image_dir, verbose=False):
        self.raw_captions = pd.read_csv(raw_captions, sep='\t')
        self.trt_image_dir = trt_image_dir
        self.pytorch_image_dir = pytorch_image_dir
        self.verbose = verbose

        # Load torchmetrics modules
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fid = FrechetInceptionDistance(feature=2048).to(self.device)
        self.clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(self.device)

    def load_image(self, image_dir, file_name):
        img = Image.open(image_dir / file_name)
        img = np.asarray(img)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        tensor = torch.Tensor(img.transpose([2, 0, 1])).to(torch.uint8)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor.unsqueeze(0).to(self.device)

    def report_accuracy(self):
        if self.verbose:
            logging.info("Accumulating results")
        for idx, file_path in enumerate(self.trt_image_dir.iterdir()):
            image_id = file_path.name[:-4]
            if self.verbose:
                logging.info(f"Image index {idx}, id {image_id}: {self.trt_image_dir / file_path.name} and {self.pytorch_image_dir / file_path.name}")
            trt_image_tensor = self.load_image(self.trt_image_dir, file_path.name)
            torch_image_tensor = self.load_image(self.pytorch_image_dir, file_path.name)
            caption = self.raw_captions.query(f"id == {image_id}")["caption"].values[0]

            self.fid.update(torch_image_tensor, real=True)
            self.fid.update(trt_image_tensor, real=False)
            self.clip.update(trt_image_tensor, caption)

        fid_score = float(self.fid.compute().item())
        clip_score = float(self.clip.compute().item())

        logging.info(f"[FID] {fid_score}, [CLIP] {clip_score}")


def main():
    # To run the accuracy tester:
    # python3 -m code.stable-diffusion-xl.tensorrt.internal.accuracy --trt-image-dir=./build/sdxl.infer.images/ --pytorch-image-dir=./build/sdxl.infer.images.ref/
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trt-image-dir",
                        help="Specify where the SDXL TRT generated images are",
                        required=True)
    parser.add_argument("--pytorch-image-dir",
                        help="Specify where the SDXL Torch reference images are",
                        required=True)
    parser.add_argument("--caption-path",
                        help="Specify where the caption file is",
                        default="build/data/coco/SDXL/coco2014/captions/captions.tsv")
    parser.add_argument("--verbose",
                        help="verbose output",
                        action="store_true")
    args = parser.parse_args()
    accuracy_tester = CoCoAccuracyTester(Path(args.caption_path), Path(args.trt_image_dir), Path(args.pytorch_image_dir), args.verbose)
    accuracy_tester.report_accuracy()


if __name__ == "__main__":
    main()
