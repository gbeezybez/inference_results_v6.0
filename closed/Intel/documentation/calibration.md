## Intel MLPerf Inference Calibration and Quantization Details

### R-GAT Quantization
Model Source: https://github.com/IllinoisGraphBenchmark/IGB-Datasets/

Model Quantization: FP32 -> INT8

Implementation: /closed/Intel/src/rgat/pytorch-cpu/backend.py

### Whisper (CPU) Quantization
Model Source: https://huggingface.co/openai/whisper-large-v3

Model Quantization: BF16 -> INT8

Details: /closed/Intel/src/whisper/pytorch-cpu/scripts/run_calibration.sh

### Whisper (XPU) Quantization
Model Source: https://huggingface.co/openai/whisper-large-v3

Model Quantization: BF16 -> INT4

Details: /closed/Intel/src/whisper/pytorch-xpu/quantize.py 

### Llama3.1-8B (CPU) Quantization
Model Source: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

Model Quantization: BF16 -> INT4

Details: /closed/Intel/src/llama3.1-8b/pytorch-cpu/scripts/run_quantization.sh

### Llama3.1-8B (XPU) Quantization
Model Source: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

Model Quantization: BF16 -> INT4

Details: /closed/Intel/src/llama2-70b/pytorch-xpu/calibration/quantize_8b.sh

### Llama2-70B (XPU) Quantization
Model Source: https://huggingface.co/meta-llama/Llama-2-70b-hf

Model Quantization: FP16 -> INT4

Details: /closed/Intel/src/llama2-70b-99.9/pytorch-xpu/calibration/quantize_70b.sh
