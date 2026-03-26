import argparse, random, time, json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import openvino as ov
from openvino import Core, PartialShape

import nncf

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

VGG_RGB_MEAN = torch.tensor([123.68, 116.78, 103.94]).view(3, 1, 1)

def build_imagenet_loader(root: str, batch_size: int, workers: int, samples: int = -1, shuffle: bool = False):
    tf_vgg = transforms.Compose([
        transforms.Resize(256, interpolation=Image.BILINEAR),
        transforms.CenterCrop(224),
        transforms.PILToTensor(),
        transforms.Lambda(lambda x: x.to(torch.float32) - VGG_RGB_MEAN),
    ])

    ds = datasets.ImageNet(root=root, split="val", transform=tf_vgg)
    if samples > 0 and samples < len(ds):
        rng = random.Random(0)
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        ds = Subset(ds, idx[:samples])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=False, drop_last=False)

def pick_softmax_output(compiled: ov.CompiledModel):
    for o in compiled.outputs:
        if "softmax" in o.get_any_name().lower():
            return o
    return compiled.outputs[0]

def drop_background(probs: np.ndarray) -> np.ndarray:
    if probs.ndim == 2 and probs.shape[1] == 1001:
        return probs[:, 1:]
    return probs

def topk_accuracy_ov(model_or_path, device: str, loader: DataLoader) -> Tuple[float, float]:
    """Evaluate top-1 and top-5 accuracy using OpenVINO inference."""
    core = Core()
    compiled = core.compile_model(model_or_path, device) if isinstance(model_or_path, (str, Path)) \
            else core.compile_model(model_or_path, device)
    out = pick_softmax_output(compiled)

    top1_hits, top5_hits, seen = 0, 0, 0
    pbar = tqdm(loader, desc="Evaluating", unit="batch", leave=False)
    for images, labels in pbar:
        arr = images.numpy()
        probs = compiled(arr)[out]
        probs = drop_background(probs)
        
        # Top-1
        top1_preds = np.argmax(probs, axis=1)
        y = labels.numpy().astype(np.int64)
        top1_hits += int(np.sum(top1_preds == y))
        
        # Top-5
        top5_preds = np.argsort(probs, axis=1)[:, -5:]
        top5_hits += int(np.sum(top5_preds == y[:, np.newaxis]))
        
        seen += y.shape[0]
        pbar.set_postfix(top1=f"{(top1_hits/seen)*100:.2f}%", top5=f"{(top5_hits/seen)*100:.2f}%")
    pbar.close()
    
    top1_acc = (top1_hits / max(1, seen)) * 100.0
    top5_acc = (top5_hits / max(1, seen)) * 100.0
    return top1_acc, top5_acc

def quantize_with_nncf(fp32_model: ov.Model,
                       calib_loader: DataLoader,
                       input_name: str,
                       subset_size: int) -> ov.Model:

    def transform_fn(data_item: Tuple[torch.Tensor, torch.Tensor]):
        images, _ = data_item
        return {input_name: images.numpy()}

    calib_dataset = nncf.Dataset(calib_loader, transform_fn)

    return nncf.quantize(
            model=fp32_model,
            calibration_dataset=calib_dataset,
            preset=nncf.QuantizationPreset.PERFORMANCE,
            subset_size=subset_size,
            fast_bias_correction=True,
    )

def main():
    ap = argparse.ArgumentParser(
        description="NNCF INT8 quantization script for ResNet-50 (MLPerf Inference).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--model", "-m",
        required=True,
        metavar="PATH",
        help="Path to OpenVINO IR model (.xml) converted with: ovc --input '[1,3,224,224]' --output softmax_tensor",
    )
    ap.add_argument(
        "--imagenet-root", "-i",
        required=True,
        metavar="DIR",
        help="Path to ImageNet dataset directory (containing ILSVRC2012_img_val.tar and ILSVRC2012_devkit_t12.tar.gz)",
    )
    ap.add_argument(
        "--out-dir", "-o",
        default="./nncf_int8_out",
        metavar="DIR",
        help="Output directory for quantized models",
    )
    ap.add_argument(
        "--device", "-d",
        default="CPU",
        choices=["CPU", "GPU", "NPU"],
        help="OpenVINO device for inference during calibration and accuracy evaluation",
    )
    ap.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        metavar="N",
        help="Batch size for inference",
    )
    ap.add_argument(
        "--workers", "-j",
        type=int,
        default=4,
        metavar="N",
        help="Number of data loader workers",
    )
    ap.add_argument(
        "--calib-subset",
        type=int,
        default=512,
        metavar="N",
        help="Number of images for NNCF calibration (more samples = better accuracy, slower)",
    )
    ap.add_argument(
        "--eval-samples",
        type=int,
        default=512,
        metavar="N",
        help="Number of images for accuracy evaluation (use 50000 for full validation set)",
    )
    args = ap.parse_args()

    core = Core()
    model_path = Path(args.model)
    assert model_path.exists(), f"Model not found: {model_path}"

    fp32_model = core.read_model(str(model_path))
    inp = fp32_model.input(0)
    pshape = inp.partial_shape

    if pshape.rank.is_dynamic or pshape[0].is_dynamic:
        try:
            fp32_model.reshape({inp: PartialShape([1,3,224,224])})
        except Exception:
            pass

    val_loader = build_imagenet_loader(args.imagenet_root, args.batch_size, args.workers, samples=args.eval_samples, shuffle=False)
    calib_loader = build_imagenet_loader(args.imagenet_root, args.batch_size, args.workers, samples=args.calib_subset, shuffle=False)

    print("\n[ Accuracy ] FP32 accuracy check in progress...")
    t0 = time.time()
    fp32_top1, fp32_top5 = topk_accuracy_ov(fp32_model, args.device, val_loader)
    print(f"[ Accuracy ] FP32 Top-1: {fp32_top1:.4f}% | Top-5: {fp32_top5:.4f}% (took {time.time()-t0:.2f}s)")

    print("\n[ Quantization ] INT8 quantization in progress...")
    input_name = fp32_model.input(0).get_any_name()
    int8_model = quantize_with_nncf(fp32_model, calib_loader, input_name, subset_size=args.calib_subset)

    try:
        out0 = int8_model.output(0)
        out0.get_node().set_friendly_name("softmax_tensor")
        out0.set_names({"softmax_tensor"})
    except Exception:
        pass

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    int8_out = out_dir / f"{model_path.stem}.xml"
    ov.save_model(int8_model, str(int8_out))
    print(f"\n[ Saved ] INT8 model saved to: {int8_out}")

    print("\n[ Accuracy ] INT8 accuracy check in progress...")
    t1 = time.time()
    int8_top1, int8_top5 = topk_accuracy_ov(int8_model, args.device, val_loader)
    print(f"[ Accuracy ] INT8 Top-1: {int8_top1:.4f}% | Top-5: {int8_top5:.4f}% (took {time.time()-t1:.2f}s)")

    ref_top1 = 76.456
    print("\n[ Summary ]")
    print(f"  {'':26} {' Top-1':>10}   {' Top-5':>10}")
    print(f"  {'Reference (MLPerf):':<26} {ref_top1:>9.4f}%   {'N/A':>10}")
    print(f"  {'FP32 OpenVINO:':<26} {fp32_top1:>9.4f}%   {fp32_top5:>9.4f}%")
    print(f"  {'INT8 OpenVINO:':<26} {int8_top1:>9.4f}%   {int8_top5:>9.4f}%")
    print(f"  {'INT8 drop vs reference:':<26} {ref_top1 - int8_top1:>+9.4f}%   {'N/A':>10}")

if __name__ == "__main__":
    main()
