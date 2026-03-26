# MangoBoost MLPerf Inference V6.0

This folder contains the detailed instructions to reproduce our following MLPerf submissions:
 1. **Single-node** (8X MI300X) *Offline* and *Server* Scenerios
 2. **Single-node** (8X MI325X) *Offline* and *Server* Scenerios

The following steps outline the processes of setting up a Docker environment and the details to reproduce our MLPerf V6.0 inference results.   

---

## 1. Preparation Before Benchmarking

### 1.1 Model and Dataset Preparation

Please download the model and dataset according to AMD's guideline [link](https://rocm.blogs.amd.com/artificial-intelligence/reproducing-amd-mlperf-inference-submission/README.html). 

### 1.2 LLMBoost Docker Preparation

Our **feature-restricted** docker is available in DockerHub, please pull our docker using the following command:
```bash
docker pull llmboost/mb-llmboost:mlperf-6.0
```
> ***Important Note***: This is a **feature-restricted** docker of our software stack [Mango LLMBoost](https://www.mangoboost.io/products/software/mango-llmboost-tm), which only enables the functionality and performance on MLPerf llama2-70B inference benchmarking. To unlock a full version of LLMBoost, please contact MangoBoost support at [contact@mangoboost.io](contact@mangoboost.io)!

Then, please use this command to run the docker container:

```bash
docker run -it --rm \
    --network host \
    --group-add video \
    --ipc host \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/dri:/dev/dri \
    --device=/dev/kfd:/dev/kfd \
    -v <path to quantized llama2-70b models>:/models/amd2025_model/model/llama2-70b-chat-hf/quantized \
    -v <path to the processed llama2-70b dataset>:/models/amd2025_model/data/processed-openorca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl \
    llmboost/mb-llmboost:mlperf-6.0
```

***From here on, we assume all the belowing commands run within the LLMBoost docker container.***

---

## 2. Single Node Benchmarking:

We begin with benchmarking on the single node setup. We will divide this section based on *Offline* and *Server* Scenarios.

### 2.1 Single-Node *Offline* Scenario Benchmarking (MI300X)

There are three steps within the benchmarking: 1. ***Performance run***, 2. ***Accuracy test***, and 3. ***Audit test***. Although we mainly focus on the performance run, we still need to run accuracy and audit test to validate our results and get a valid official submission. 

At the beginning please start the LLMBoost service by running the command:
```bash
cd /workspace/apps/mlperf
python3 server.py --test_mode Offline --model_path "/models/amd2025_model/model/llama2-70b-chat-hf/quantized" --accelerator_name mi300x --load_balancing_mode shortest_first
```

When the service is started, you will see LLMBoost is listening on two ports: `0.0.0.0:8000` and `0.0.0.0:8001`. You do not need to kill and restart the service unless your whole benchmarking is finished.

### 2.1.1 Single-Node *Offline* Performance Run

With the LLMBoost service on, you will need to start another ternimal and go into the docker container for benchmarking. Here is the command for the performance run:
```bash
SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYSTEM_NAME=${SYSTEM_NAME:-"8xMI300X_2xEPYC_9534"}
cd /workspace/apps/mlperf
python3 client.py \
    --test_mode Offline \
    --user_conf conf/user_llama2-70b_8x_mi300x.conf \
    --sut_server_addr "http://localhost:8000" \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/performance/run_1
```
The benchmarking will run one hour for maximizing the performance. You can also modify on the user config file (e.g. `conf/user_llama2-70b_8x_mi300x.conf` for 8xmi300x) to lower down the duration, especially when you just want a quick try.

***Expected Performance:***

| System | 10-minute run | 1-hour run |
|------|------|------|
| Single-Node MI300X (8x MI300X) | >26k tokens/s | >27k tokens/s |

### 2.1.2 Single-Node *Offline* Accuracy Test

For an official MLPerf [closed-division](https://mlcommons.org/benchmarks/inference-datacenter/#:~:text=Offline-,Divisions,-MLPerf%20aims%20to) submission, we need to make sure the model accuracy reaches a certain accuracy requirement. To run the accuracy test, you can run the following commands:
```bash
SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYSTEM_NAME=${SYSTEM_NAME:-"8xMI300X_2xEPYC_9534"}
python3 client.py \
    --test_mode Offline \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/accuracy  \
    --user_conf conf/user_llama2-70b_8x_mi300x.conf  \
    --accuracy_test \
    --sut_server_addr="http://localhost:8000"
bash tools/check_llama2_accuracy_scores.sh $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/accuracy/mlperf_log_accuracy.json
cat $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/accuracy/accuracy.txt
```
The accuracy test will send 24576 sampels to the LLMBoost services and measure the accuracy of respones. In the end, it will generate a set of rouge-score. PLEASE make sure the accuracy is above the following requirements so that it can pass the submission checker:
```
# reference accuracy number, please get number greater than these
(99% metric): {rouge1: 44.39, rouge2: 22.01, rougeL: 28.59}
```

### 2.1.3 Single-Node *Offline* Audit Test

The Audit test is to make sure the systems is compliance with the rules. Please run it with the following command:
```bash
SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYSTEM_NAME=${SYSTEM_NAME:-"8xMI300X_2xEPYC_9534"}
TEST06_DIR=/workspace/apps/mlperf/tools/compliance/nvidia/TEST06
cp $TEST06_DIR/audit.config ./
python3 client.py \
    --test_mode Offline \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/audit/compliance/TEST06  \
    --sut_server_addr="http://localhost:8000"
python3 $TEST06_DIR/run_verification.py -c $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/audit/compliance/TEST06 -o $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/audit/compliance -s Offline
rm audit.config
```
The audit test only sends 100 samples to LLMBoost services, which will be really quick. You can check whether it pass or fail according to the output file `$SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Offline/audit/compliance/TEST06/verify_accuracy.txt`. The expected output is:
```bash
First token check pass: Skipped
EOS check pass: True
Sample length check pass: True
TEST06 verification complete
```

### 2.2 Single-Node *Server* Scenario Benchmarking

In the *Server* Scenario, it contains the same steps: ***Performance run***, ***Accuracy test***, and ***Audit test***.  

At the beginning please start the LLMBoost service by running the command:
```bash
cd /workspace/apps/mlperf
python3 server.py --test_mode Server --model_path "/models/amd2025_model/model/llama2-70b-chat-hf/quantized" --accelerator_name mi300x --load_balancing_mode shortest_first
```

### 2.2.1 Single-Node *Server* Performance Run

With the LLMBoost service on, you will need to start another ternimal and go into the docker container for benchmarking. Here is the command for the performance run:
```bash
SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYSTEM_NAME=${SYSTEM_NAME:-"8xMI300X_2xEPYC_9534"}
cd /workspace/apps/mlperf
python3 client.py \
    --test_mode Server \
    --user_conf conf/user_llama2-70b_8x_mi300x.conf \
    --sut_server_addr "http://localhost:8000" \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/performance/run_1
```
You might see the benchmarking results to be `INVALID`, which is usually because the TTFT and TPOT performance doesn't meet the [constraints](https://mlcommons.org/2024/03/mlperf-llama2-70b/#:~:text=Latency%20constraints%20for%20the%20server%20scenario) (this requirement is only in *Server* Scenario but not in *Offline* Scenario). If you encounter this issue, you will need to modify user config file (e.g. `conf/user_llama2-70b_8x_mi300x.conf` for 8xmi300x) to lower down the `llama2-70b.Server.target_qps`.

Same as *Offline* Scenario, you can also lower down the benchmarking duration to 10 minutes for a quick try, and only do one-hour benchmarking in your final confident benchmark to maximize the performance.

***Expected Performance:***

| System | 10-minute run | 1-hour run |
|------|------|------|
| Single-Node MI300X (8x MI300X) | >25k tokens/s | >25.5k tokens/s |

### 2.2.2 Single-Node *Server* Accuracy Test

For a valid MLPerf [closed-division](https://mlcommons.org/benchmarks/inference-datacenter/#:~:text=Offline-,Divisions,-MLPerf%20aims%20to) submission, we need to make sure the model accuracy reaches a certain accuracy requirement. To run the accuracy test, you can run the following commands:
```bash
SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYSTEM_NAME=${SYSTEM_NAME:-"8xMI300X_2xEPYC_9534"} 
python3 client.py \
    --test_mode Server \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/accuracy  \
    --user_conf conf/user_llama2-70b_8x_mi300x.conf  \
    --accuracy_test \
    --sut_server_addr="http://localhost:8000"
bash tools/check_llama2_accuracy_scores.sh $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/accuracy/mlperf_log_accuracy.json
cat $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/accuracy/accuracy.txt
```
PLEASE make sure the accuracy is above the same requirements shown in previous *offline* accuracy test section so that it can pass the submission checker.

### 2.2.3 Single-Node *Server* Audit Test

The Audit test is to make sure the systems is compliance with the rules. Please run it with the following command:
```bash
SUBMISSION_DIR=/workspace/apps/mlperf/submission
SYSTEM_NAME=${SYSTEM_NAME:-"8xMI300X_2xEPYC_9534"}
TEST06_DIR=/workspace/apps/mlperf/tools/compliance/nvidia/TEST06
cp $TEST06_DIR/audit.config ./
python3 client.py \
    --test_mode Server \
    --result_dir $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/audit/compliance/TEST06  \
    --sut_server_addr="http://localhost:8000"
python3 $TEST06_DIR/run_verification.py -c $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/audit/compliance/TEST06 -o $SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/audit/compliance -s Server
rm audit.config
```
You can check whether it pass or fail according to the output file `$SUBMISSION_DIR/results/llama2-70b/$SYSTEM_NAME/Server/audit/compliance/TEST06/verify_accuracy.txt`. 

---

## 3. Single-Node Benchmarking (MI325X)
Different from all other MI300X setups, for MI325X single-node llama2-70B benchmarking, we specifically used AMD harness code to generate the results we submitted in mlperf v6.0 inference. To reproduce this result, please follow the command from AMD.

---

## 4. Packing the Submission & Conducting Validation Checking

After the commands above, all the results are ready to be packaged. Please run the commands below to package the results.

```bash
cd /workspace/apps/mlperf
bash submission_package.sh
```