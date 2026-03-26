# MLPerf Inference Scaleout - Reproduction Commands

## DeepSeek-R1

### GB200x72

```bash
# Offline
salloc --nodes=18 --partition=gb200
./scaleout/run_scaleout.sh --stage all --atomic-system GB200-NVL72_GB200-186GB_aarch64x8 --gpus-per-node 4 --dp-multiplicity 9 --run-args "--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint"

# Server
salloc --nodes=18 --partition=gb200
./scaleout/run_scaleout.sh --stage all --atomic-system GB200-NVL72_GB200-186GB_aarch64x8 --gpus-per-node 4 --dp-multiplicity 9 --run-args "--benchmarks=deepseek-r1 --scenarios=Server --core_type=trtllm_endpoint"

# Interactive (WIP)
```

### GB300x72

```bash
# Offline
salloc --nodes=18 --partition=gb300
./scaleout/run_scaleout.sh --stage all --atomic-system GB300-NVL72_GB300-288GB_aarch64x8 --gpus-per-node 4 --dp-multiplicity 9 --run-args "--benchmarks=deepseek-r1 --scenarios=Offline --core_type=trtllm_endpoint"

# Server
salloc --nodes=18 --partition=gb300
./scaleout/run_scaleout.sh --stage all --atomic-system GB300-NVL72_GB300-288GB_aarch64x8 --gpus-per-node 4 --dp-multiplicity 9 --run-args "--benchmarks=deepseek-r1 --scenarios=Server --core_type=trtllm_endpoint"

# Interactive (WIP)
```

## Llama3.1-405b

### GB200x72

```bash
# Offline
salloc --nodes=18 --partition=gb200
./scaleout/run_scaleout.sh --stage all --atomic-system GB200-NVL72_GB200-186GB_aarch64x4 --gpus-per-node 4 --dp-multiplicity 18 --run-args "--benchmarks=llama3_1-405b --scenarios=Offline --core_type=trtllm_endpoint"
# Server
salloc --nodes=18 --partition=gb200
#IFB:
#./scaleout/run_scaleout.sh --stage all --atomic-system GB200-NVL72_GB200-186GB_aarch64x4 --gpus-per-node 4 --dp-multiplicity 18 --run-args "--benchmarks=llama3_1-405b --scenarios=Server --core_type=trtllm_endpoint"
#Disagg (better perf)
./scaleout/run_scaleout_disagg.py \
    --stage all \
    --gpus-per-node 4 \
    --num-ctx-servers 14 \
    --num-gen-servers 2 \
    --ctx-atomic-system GB200-NVL72_GB200-186GB_aarch64x4 \
    --gen-atomic-system GB200-NVL72_GB200-186GB_aarch64x8 \
    --harness-system GB200-NVL72_GB200-186GB_aarch64x72 \
    --ctx-run-args "--benchmarks=llama3_1-405b --scenarios=Server --core_type=trtllm_endpoint --config_id=ctx" \
    --gen-run-args "--benchmarks=llama3_1-405b --scenarios=Server --core_type=trtllm_endpoint" \
    --harness-run-args "--benchmarks=llama3_1-405b --scenarios=Server --core_type=trtllm_endpoint --config_id=disagg" 

# Interactive
salloc --nodes=18 --partition=gb200
python3 ./scaleout/run_scaleout_disagg.py \
    --stage all \
    --gpus-per-node 4 \
    --num-ctx-servers 14 \
    --num-gen-servers 2 \
    --ctx-atomic-system GB200-NVL72_GB200-186GB_aarch64x4 \
    --gen-atomic-system GB200-NVL72_GB200-186GB_aarch64x8 \
    --harness-system GB200-NVL72_GB200-186GB_aarch64x72 \
    --ctx-run-args "--benchmarks=llama3_1-405b --scenarios=Interactive --core_type=trtllm_endpoint " \
    --gen-run-args "--benchmarks=llama3_1-405b --scenarios=Interactive --core_type=trtllm_endpoint " \
    --harness-run-args "--benchmarks=llama3_1-405b --scenarios=Interactive --core_type=trtllm_endpoint "
```

### GB300x72

```bash
# Offline
salloc --nodes=18 --partition=gb300
./scaleout/run_scaleout.sh --stage all --atomic-system GB300-NVL72_GB300-288GB_aarch64x2 --gpus-per-node 4 --dp-multiplicity 36 --run-args "--benchmarks=llama3_1-405b --scenarios=Offline --core_type=trtllm_endpoint"

# Server
salloc --nodes=18 --partition=gb300
./scaleout/run_scaleout.sh --stage all --atomic-system GB300-NVL72_GB300-288GB_aarch64x2 --gpus-per-node 4 --dp-multiplicity 36 --run-args "--benchmarks=llama3_1-405b --scenarios=Server --core_type=trtllm_endpoint"

# Interactive (WIP)
salloc --nodes=18 --partition=gb300
./scaleout/run_scaleout_disagg.py \
    --stage all \
    --gpus-per-node 4 \
    --num-ctx-servers 26 \
    --num-gen-servers 5 \
    --ctx-atomic-system GB300-NVL72_GB300-288GB_aarch64x2 \
    --gen-atomic-system GB300-NVL72_GB300-288GB_aarch64x4 \
    --harness-system GB300-NVL72_GB300-288GB_aarch64x72 \
    --ctx-run-args "--benchmarks=llama3_1-405b --scenarios=Interactive --core_type=trtllm_endpoint" \
    --gen-run-args "--benchmarks=llama3_1-405b --scenarios=Interactive --core_type=trtllm_endpoint" \
    --harness-run-args "--benchmarks=llama3_1-405b --scenarios=Interactive --core_type=trtllm_endpoint" 
```

## Llama2-70b

### GB200x72

```bash
# Offline
salloc --nodes=18 --partition=gb200
./scaleout/run_scaleout.sh --stage all --atomic-system GB200-NVL72_GB200-186GB_aarch64x1 --gpus-per-node 4 --dp-multiplicity 72 --run-args "--benchmarks=llama2-70b --scenarios=Offline --core_type=trtllm_endpoint"

# Server
salloc --nodes=18 --partition=gb200
python3 ./scaleout/run_scaleout_disagg.py \
    --stage all \
    --gpus-per-node 4 \
    --num-ctx-servers 28 \
    --num-gen-servers 44 \
    --num-master-servers 9 \
    --ctx-atomic-system GB200-NVL72_GB200-186GB_aarch64x1 \
    --gen-atomic-system GB200-NVL72_GB200-186GB_aarch64x1 \
    --ctx-run-args "--benchmarks=llama2-70b --scenarios=Server --core_type=trtllm_endpoint --config_id=ctx_config_x72" \
    --gen-run-args "--benchmarks=llama2-70b --scenarios=Server --core_type=trtllm_endpoint --config_id=gen_config_x72" \
    --harness-run-args "--benchmarks=llama2-70b --scenarios=Server --core_type=trtllm_endpoint"

# Interactive
salloc --nodes=18 --partition=gb200
python3 ./scaleout/run_scaleout_disagg.py \
    --stage all \
    --gpus-per-node 4 \
    --num-ctx-servers 28 \
    --num-gen-servers 44 \
    --num-master-servers 9 \
    --ctx-atomic-system GB200-NVL72_GB200-186GB_aarch64x1 \
    --gen-atomic-system GB200-NVL72_GB200-186GB_aarch64x1 \
    --ctx-run-args "--benchmarks=llama2-70b --scenarios=Interactive --core_type=trtllm_endpoint --config_id=ctx_config_x72" \
    --gen-run-args "--benchmarks=llama2-70b --scenarios=Interactive --core_type=trtllm_endpoint --config_id=gen_config_x72" \
    --harness-run-args "--benchmarks=llama2-70b --scenarios=Interactive --core_type=trtllm_endpoint"
```

### GB300x72

```bash
# Offline
salloc --nodes=18 --partition=gb300
./scaleout/run_scaleout.sh --stage all --atomic-system GB300-NVL72_GB300-288GB_aarch64x1 --gpus-per-node 4 --dp-multiplicity 72 --run-args "--benchmarks=llama2-70b --scenarios=Offline --core_type=trtllm_endpoint"

# Server
salloc --nodes=18 --partition=gb300
python3 ./scaleout/run_scaleout_disagg.py \
    --stage all \
    --gpus-per-node 4 \
    --num-ctx-servers 28 \
    --num-gen-servers 44 \
    --num-master-servers 9 \
    --ctx-atomic-system GB300-NVL72_GB300-288GB_aarch64x1 \
    --gen-atomic-system GB300-NVL72_GB300-288GB_aarch64x1 \
    --ctx-run-args "--benchmarks=llama2-70b --scenarios=Server --core_type=trtllm_endpoint --config_id=ctx_config_x72" \
    --gen-run-args "--benchmarks=llama2-70b --scenarios=Server --core_type=trtllm_endpoint --config_id=gen_config_x72" \
    --harness-run-args "--benchmarks=llama2-70b --scenarios=Server --core_type=trtllm_endpoint"

# Interactive
salloc --nodes=18 --partition=gb300
python3 ./scaleout/run_scaleout_disagg.py \
    --stage all \
    --gpus-per-node 4 \
    --num-ctx-servers 28 \
    --num-gen-servers 44 \
    --num-master-servers 9 \
    --ctx-atomic-system GB300-NVL72_GB300-288GB_aarch64x1 \
    --gen-atomic-system GB300-NVL72_GB300-288GB_aarch64x1 \
    --ctx-run-args "--benchmarks=llama2-70b --scenarios=Interactive --core_type=trtllm_endpoint --config_id=ctx_config_x72" \
    --gen-run-args "--benchmarks=llama2-70b --scenarios=Interactive --core_type=trtllm_endpoint --config_id=gen_config_x72" \
    --harness-run-args "--benchmarks=llama2-70b --scenarios=Interactive --core_type=trtllm_endpoint"
```

## GPT-OSS-120b

### GB200x72

```bash
# Offline
salloc --nodes=18 --partition=gb200
./scaleout/run_scaleout.sh --stage all --atomic-system GB200-NVL72_GB200-186GB_aarch64x1 --gpus-per-node 4 --dp-multiplicity 72 --run-args "--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint"

# Server
salloc --nodes=18 --partition=gb200
./scaleout/run_scaleout.sh --stage all --atomic-system GB200-NVL72_GB200-186GB_aarch64x1 --gpus-per-node 4 --dp-multiplicity 72 --run-args "--benchmarks=gpt-oss-120b --scenarios=Server --core_type=trtllm_endpoint"

# Interactive
salloc --nodes=18 --partition=gb200
python3 ./scaleout/run_scaleout_disagg.py \
    --stage all \
    --gpus-per-node 4 \
    --num-ctx-servers 24 \
    --num-gen-servers 12 \
    --num-master-servers 12 \
    --ctx-atomic-system GB200-NVL72_GB200-186GB_aarch64x1 \
    --gen-atomic-system GB200-NVL72_GB200-186GB_aarch64x4 \
    --ctx-run-args "--benchmarks=gpt-oss-120b --scenarios=Interactive --core_type=trtllm_endpoint" \
    --gen-run-args "--benchmarks=gpt-oss-120b --scenarios=Interactive --core_type=trtllm_endpoint" \
    --harness-run-args "--benchmarks=gpt-oss-120b --scenarios=Interactive --core_type=trtllm_endpoint"
```

### GB300x72

```bash
# Offline
salloc --nodes=18 --partition=gb300
./scaleout/run_scaleout.sh --stage all --atomic-system GB300-NVL72_GB300-288GB_aarch64x1 --gpus-per-node 4 --dp-multiplicity 72 --run-args "--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint"

# Server
salloc --nodes=18 --partition=gb300
./scaleout/run_scaleout.sh --stage all --atomic-system GB300-NVL72_GB300-288GB_aarch64x1 --gpus-per-node 4 --dp-multiplicity 72 --run-args "--benchmarks=gpt-oss-120b --scenarios=Server --core_type=trtllm_endpoint"

# Interactive
salloc --nodes=18 --partition=gb300
python3 ./scaleout/run_scaleout_disagg.py \
    --stage all \
    --gpus-per-node 4 \
    --num-ctx-servers 20 \
    --num-gen-servers 13 \
    --num-master-servers 12 \
    --ctx-atomic-system GB300-NVL72_GB300-288GB_aarch64x1 \
    --gen-atomic-system GB300-NVL72_GB300-288GB_aarch64x4 \
    --ctx-run-args "--benchmarks=gpt-oss-120b --scenarios=Interactive --core_type=trtllm_endpoint" \
    --gen-run-args "--benchmarks=gpt-oss-120b --scenarios=Interactive --core_type=trtllm_endpoint" \
    --harness-run-args "--benchmarks=gpt-oss-120b --scenarios=Interactive --core_type=trtllm_endpoint"
```
