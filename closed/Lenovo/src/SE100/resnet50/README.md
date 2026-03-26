# ResNet-50 OpenVINO Implementation

This directory contains the OpenVINO-based MLPerf Inference implementation for ResNet-50.
These files are designed as patch-replacements for the reference implementation.

## Directory Structure
```
resnet50/
├── python/              # Python backend (patch-replace)
│   ├── backend_openvino.py  # OpenVINO inference backend
│   └── main.py              # MLPerf LoadGen harness
├── nncf-quantize/       # INT8 quantization scripts
│   ├── quantize_resnet50.py # NNCF quantization script
│   ├── convert_model.sh     # ONNX to OpenVINO IR conversion
│   └── quantize_model.sh    # Run quantization
├── scripts/             # Setup and utility scripts
├── run_mlperf_ov_resnet.sh  # Main benchmark runner
├── user.conf            # MLPerf user configuration
└── requirements.txt     # Python dependencies
```

## Setup
```bash
./scripts/install_prerequisites.sh
./scripts/setup_env.sh
./scripts/download_model.sh
./nncf-quantize/convert_model.sh
./nncf-quantize/quantize_model.sh
./scripts/prepare_imagenet_dataset.sh
```

## Run Benchmark
```bash
# Performance
./run_mlperf_ov_resnet.sh -t performance -d CPU -p INT8 -s SingleStream

# Accuracy
./run_mlperf_ov_resnet.sh -t accuracy -d CPU -p INT8 -s SingleStream
```

## Run Compliance Tests
```bash
./run_mlperf_ov_resnet.sh -t performance -d CPU -p INT8 -s SingleStream --audit TEST01
./run_mlperf_ov_resnet.sh -t performance -d CPU -p INT8 -s SingleStream --audit TEST04
```

## Dependencies
- OpenVINO >= 2024.0
- NNCF (for quantization)
- MLPerf LoadGen
