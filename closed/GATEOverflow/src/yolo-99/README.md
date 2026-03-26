## 2D Object Detection using YOLOv11

In this submission round, **GATEOverflow** utilized the **MLCommons Reference implementation (PyTorch)** to run MLPerf inference on the **YOLOv11** model. This submission adheres to the **Closed Division** accuracy metric for edge devices, encompassing three key scenarios: **SingleStream**, **MultiStream**, and **Offline**.

The submission specifically targets the **YOLO-99** quality threshold, ensuring that inference results maintain at least 99% of the reference accuracy target (**53.4% Mean Average Precision**).

## Implementation Details

The benchmark was executed using [MLCFlow automation](https://github.com/mlcommons/mlcflow) to ensure a reproducible environment:
- GPU: Runs were performed within a Docker container.
- CPU: Runs were performed natively on the host system.

Detailed system specifications, including hardware configurations and library versions, are documented in the accompanying system JSON file.


## Run Instructions

To reproduce these results, please refer to the official [**MLPerf Inference: Object Detection (YOLO) Guide**](https://docs.mlcommons.org/inference/benchmarks/object_detection/yolo/#__tabbed_24_1). This guide provides detailed steps for the Docker container setup, dependency installation, and the specific MLCFlow commands required for the run.