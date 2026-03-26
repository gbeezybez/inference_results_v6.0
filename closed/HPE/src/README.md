To reproduce HPE implementations for a specific NVIDIA Tensor-RT version, copy the required src files to closed/HPE/code/.

To run with **10 GPUs per node** using **TensorRT 10.14**, set `gpus_per_node` in
`closed/HPE/3rdparty/trtllm/tensorrt_llm/commands/build.py`:

```python 
model_config.mapping.gpus_per_node = 10
```