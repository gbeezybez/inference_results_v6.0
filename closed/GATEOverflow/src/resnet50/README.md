## Image classification with ResNet50

In this submission round, **GATEOverflow** utilized the **Nvidia implementation (TensorRT)** to run MLPerf inference on the **ResNet50** model.  This submission adheres to the **Closed Division** accuracy metric for edge devices, encompassing three key scenarios: **SingleStream**, **MultiStream**, and **Offline**.

The submission specifically targets the **ResNet50** quality threshold, ensuring that inference results maintain at least 99% of the reference accuracy target (**76.46% Top-1 Accuracy**).

The implementation leverages [MLCFlow automation](https://github.com/mlcommons/mlcflow) within a **Docker environment** to provide a standardized, scalable, and reproducible benchmarking environment. Detailed system specifications, including hardware configurations and library versions, are documented in the accompanying system JSON file.


## Run Instructions

To reproduce these results, please refer to the official [**MLPerf Inference: Image Classification (ResNet50) Guide**](https://docs.mlcommons.org/inference/benchmarks/image_classification/resnet50/). This guide provides detailed steps for the Docker container setup, dependency installation, and the specific MLCFlow commands required for the run.