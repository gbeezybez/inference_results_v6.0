# MLPerf Inference v6.0 - Open - Krai

To run experiments individually, use the following commands.

## h200_n1-vllm - llama3_1-70b-fp8_dyn - server

### Accuracy  
```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=AccuracyOnly,loadgen_scenario=Server,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_loadgen_workers=1,tp=1,pp=1,dp=8,num_gpus=8,quantization=fp8,gpu_memory_utilization=0.95,model_family=llama3_1,model_variant=70b,loadgen_target_qps=93,collection_name=results_server_040226,num_openai_workers=17,openai_max_connections=1000,max_num_seqs=858,max_num_batched_tokens=36498,sut_name=h200_n1-vllm,server_docker_image_tag=nightly-d88a1df699f68e5284fe3a3170f8ae292a3e9c3f
```

### Performance 
```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Server,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_loadgen_workers=1,tp=1,pp=1,dp=8,num_gpus=8,quantization=fp8,gpu_memory_utilization=0.95,model_family=llama3_1,model_variant=70b,loadgen_target_qps=85,collection_name=results_server_040226,num_openai_workers=17,openai_max_connections=1000,max_num_seqs=858,max_num_batched_tokens=36498,sut_name=h200_n1-vllm,server_docker_image_tag=nightly-d88a1df699f68e5284fe3a3170f8ae292a3e9c3f
```
