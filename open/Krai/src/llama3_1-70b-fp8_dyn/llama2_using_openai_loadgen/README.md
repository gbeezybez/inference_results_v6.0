# MLPerf Inference - Llama2 task

## Prerequisites
* Docker must be installed.
* The user running benchmarks must be incuded in the `docker` group.

## Define a local workspace directory
```
export WORKSPACE_DIR=/workspace
mkdir -p ${WORKSPACE_DIR}
mkdir -p ${WORKSPACE_DIR}/${USER}
```

## Install KRAI [AXS](https://github.com/krai/axs)

### Clone

Clone the AXS repository under `${WORKSPACE_DIR}/${USER}`:
```
git clone https://github.com/krai/axs ${WORKSPACE_DIR}/${USER}/axs
```

### Init

Define environment variables in your `~/.bashrc`:
```
echo "

# AXS.
export WORKSPACE_DIR=/workspace
export PATH=${WORKSPACE_DIR}/${USER}/axs:${PATH}
export AXS_WORK_COLLECTION=${WORKSPACE_DIR}/${USER}/work_collection

" >> ~/.bashrc
```

### Test
```
source ~/.bashrc
axs version
```

## Import public AXS repositories

Import the required public repos into your work collection:

```
axs byquery git_repo,collection,repo_name=axs2mlperf
axs byquery git_repo,collection,repo_name=axs2kiss
```

## Download artifacts

Use a [HuggingFace access token](https://huggingface.co/docs/hub/en/security-tokens) (`export HF_TOKEN=...`) to download the model and its tokenizer.

### Model
The benchmark supports the following models:
* Llama2-70b
* Llama3.1-70b
* Llama3.3-70b
To download a desired model, set the parameter `MODEL_FAMILY` to one of `llama2`, `llama3_1` or `llama3_3` and run
```
axs byquery downloaded,hf_model,model_family=${MODEL_FAMILY},variant=70b,hf_token=${HF_TOKEN}
```

You can also provide your own model by adding `,model_path=...` to the commands below.

### Tokenizer
```
axs byquery downloaded,hf_tokeniser,model_family=${MODEL_FAMILY},variant=70b,total_samples=24576,hf_token=${HF_TOKEN}
```

### Dataset (OpenOrca)
```
axs byquery downloaded,dataset_name=openorca,model_family=llama2,variant=70b,desired_python_version===3.9
```

## Benchmark

See below example commands for the Server/Offline scenarios and under the Accuracy/Performance/Compliance modes.

See commands used for actual submission runs under the corresponding subdirectories of `measurements/`.

The `quantization` parameter support is as follows:
* vllm - it can be set to `None` if you want to use an unquantized or prequantized model or to `fp8` to apply the dynamic fp8 quantisation at model loading;
* sglang - it can be set to `None` if you want to use an unquantized model or to `fp8` to apply the dynamic fp8 quantisation on an unquantize model at model loading or to load a prequantized fp8 model;
* nim - the parameter is not used;
* trtllm - the parameter is not used.

The `backend` parameter supports options `cuda` and `rocm`.

The benchmark supports the following inference servers: vLLM (by default), SGLang, NIM and TensorRT-LLM.
To activate not default ones, specify `inference_server=sglang`, `inference_server=nim` or `inference_server=trtllm`.

### Offline

#### Accuracy
```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,backend=cuda,num_openai_workers=8,num_loadgen_workers=1,max_num_seqs=768,max_num_batched_tokens=16384,openai_max_connections=900,model_family=${MODEL_FAMILY}
```

#### Performance
```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,backend=cuda,num_openai_workers=8,num_loadgen_workers=1,max_num_seqs=768,max_num_batched_tokens=16384,openai_max_connections=900,model_family=${MODEL_FAMILY},loadgen_target_qps=<desired qps>
```

### Server

#### Accuracy
```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=AccuracyOnly,loadgen_scenario=Server,backend=cuda,num_openai_workers=8,num_loadgen_workers=1,max_num_seqs=256,max_num_batched_tokens=16384,openai_max_connections=900,model_family=${MODEL_FAMILY},loadgen_target_qps=<desired qps>
```

### Offline

#### Performance
```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Server,backend=cuda,num_openai_workers=8,num_loadgen_workers=1,max_num_seqs=256,max_num_batched_tokens=16384,openai_max_connections=900,model_family=${MODEL_FAMILY},loadgen_target_qps=<desired qps>
```
