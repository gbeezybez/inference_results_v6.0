# MLPerf Inference 6.0

## Setup

### Model and Dataset
To download model and dataset we are using `/data` directory. 
Download the dataset for the benchmark by running the below command.

```bash
bash setup/llama2-70b-99/download_dataset.sh
```

## Download and Quantize Model
If you already have download unquantized model then place it under `/data/inference/model/llama2-70b-chat-hf/orig/`.
If the model already exists: The HF token is not required and the download step will be skipped.

## To Quantize Model 

```bash
MODEL_OPTION="_fp4"
bash setup/llama2-70b-99/download_model$MODEL_OPTION.sh
```
If the model does not exist: Provide your Hugging Face token to download the model first and then quantize.

```bash
HUGGINGFACE_ACCESS_TOKEN="<your HF token goes here>"
MODEL_OPTION="_fp4"
bash setup/llama2-70b-99/download_model$MODEL_OPTION.sh $HUGGINGFACE_ACCESS_TOKEN
```

## Output Location:

Unquantized model: `/data/inference/model/llama2-70b-chat-hf/orig/`
FP4 quantized model: `/data/inference/model/llama2-70b-chat-hf/fp4_quantized/`

## Inference

### Runtime tunables

To boost the machine's performance further, execute the following script before any performance test (should be set once after a reboot):

```bash
bash setup/runtime_tunables.sh
```

### Docker

Build the docker image for the benchmark by running the below command

```bash
bash setup/llama2-70b-99/build_docker.sh
```

Start the docker container for the benchmark by running the below commands

```bash
export EXTRA_ARGS="--rm --workdir /lab-mlperf-inference/submission"
bash setup/llama2-70b-99/start_docker.sh
```

## Running Experiments for Submission

### Option 1: Run Multiple Experiments and Select Best Runs

#### 1. Run Experiments

**Offline Scenario:**

```bash
python3 submission.py --model llama2-70b-99 experiment --scenario Offline --model-conf /lab-mlperf-inference/code/llama2-70b-99/offline_mi355x.yaml --user-conf /lab-mlperf-inference/code/llama2-70b-99/user_mi355x.conf
```

**Server Scenario:**

```bash
python3 submission.py --model llama2-70b-99 experiment --scenario Server --model-conf /lab-mlperf-inference/code/llama2-70b-99/server_mi355x.yaml --user-conf /lab-mlperf-inference/code/llama2-70b-99/user_mi355x.conf
```

**Interactive Scenario:**

```bash
python3 submission.py --model llama2-70b-99 experiment --scenario Interactive --model-conf /lab-mlperf-inference/code/llama2-70b-99/interactive_mi355x.yaml --user-conf /lab-mlperf-inference/code/llama2-70b-99/user_mi355x.conf
```

#### 2. Check Status

Check the current best model state:

```bash
python3 submission.py --model llama2-70b-99 status
```

#### 3. Update Best Result

Select the current best result for a scenario:

```bash
# Offline
python3 submission.py --model llama2-70b-99 update_best --scenario Offline

# Server
python3 submission.py --model llama2-70b-99 update_best --scenario Server

# Interactive
python3 submission.py --model llama2-70b-99 update_best --scenario Interactive
```

#### 4. Prepare (Accuracy/Compliance)

**Run Accuracy:**

```bash
# Offline
python3 submission.py --model llama2-70b-99 prepare --scenario Offline accuracy

# Server
python3 submission.py --model llama2-70b-99 prepare --scenario Server accuracy

# Interactive
python3 submission.py --model llama2-70b-99 prepare --scenario Interactive accuracy
```

**Run Compliance:**

```bash
# Offline
python3 submission.py --model llama2-70b-99 prepare --scenario Offline compliance

# Server
python3 submission.py --model llama2-70b-99 prepare --scenario Server compliance

# Interactive
python3 submission.py --model llama2-70b-99 prepare --scenario Interactive compliance
```

**Force Overwrite (if result already exists):**

```bash
python3 submission.py --model llama2-70b-99 prepare --scenario Offline --force accuracy
python3 submission.py --model llama2-70b-99 prepare --scenario Offline --force compliance
```

#### 5. Package for Submission

Package the current best results (run `status` first to verify everything is ready):

```bash
# Set required environment variables first
export GPU_COUNT=8
export GPU_NAME="MI355X"
export CPU_COUNT=2
export CPU_NAME="EPYC-9575F" #Set CPU_NAME based on your hardware you use. You can use `lscpu | grep name`.
export COMPANY="AMD" #Set Your company name.

# Then package
python3 submission.py --model llama2-70b-99 package
```

> **Note:** See `.submission_package_env` for environment variable details.

### Running Partial Runs

When using the automated script (Option 1), the following flags can control which parts run:

| Flag | Default | Description |
|------|---------|-------------|
| `OFFLINE` | 1 | Run Offline scenario |
| `SERVER` | 1 | Run Server scenario |
| `PERFORMANCE` | 1 | Run performance benchmarks |
| `ACCURACY` | 1 | Run accuracy tests |
| `COMPLIANCE` | 1 | Run compliance tests |
| `PACKAGE` | 1 | Package results |

Define flags before the command to disable specific parts:

```bash
# Skip Offline and Package steps
OFFLINE=0 PACKAGE=0 bash /lab-mlperf-inference/submission/llama2_70b.sh

# Run only performance (skip accuracy and compliance)
ACCURACY=0 COMPLIANCE=0 bash /lab-mlperf-inference/submission/llama2_70b.sh

# Run only Offline scenario, skip Server
SERVER=0 bash /lab-mlperf-inference/submission/llama2_70b.sh
```
### Option 2: One short submission package

```bash
mkdir llama2_submission
COMPANY="AMD" CPU_NAME="EPYC_9575F" GPU_NAME="mi355x" GPU_COUNT=8 RESULTS="/lab-mlperf-inference/submission/llama2_submission" ENABLE_POWER_SETUP=1 bash /lab-mlperf-inference/submission/llama2_70b.sh
```
Note: Please change COMPANY and CPU_NAME as per your hardware
