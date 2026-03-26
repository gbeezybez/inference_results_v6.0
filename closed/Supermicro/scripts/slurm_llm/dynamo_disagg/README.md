# Dynamo + TensorRT-LLM Disaggregated Serving 
## Prerequisites
- SLURM allocation with `enroot` and `pyxis` support
- Container image with Dynamo + TRT-LLM
  - TODO: Add container building steps
- Prefill config YAML (`prefill_config.yaml`)
- Decode config YAML (`decode_config.yaml`)
- To benchmark via `make run_harness`, please set `MLPINF_HTTP_USE_COMPLETIONS=1` in your env

## Terminology
### 1. Prefill/Decode Worker
A single `trtllm-serve` instance configured by dynamo to run compute-bound prefill or memory-bound decode.
### 2. Frontend server
A combination of NATS server, ETCD server and Dynamo router that manages routing of requests between workers. 

### 3. Disagg Cluster
`M` prefill workers + `N` decode workers + `1` frontend server. 
- Each disagg cluster has it's own unique `v1/chat/completions` endpoint that harness can target.
- 1:1 mapping between clusters <-> frontends 

## TLDR - Launch a generic disagg cluster with `N` prefill, `M` decode workers
Starting with a trtllm config yaml file for each the prefill and decode workers, you may stand up a disagg cluster using the following steps. 

### 1. Write the disagg cluster yml file

Create a `disagg_cluster.yml` file as follows. Here, we are launching 1 prefill server, and 4 decode servers across 72 GPUs.
```yaml
benchmark: deepseek-r1
scenario: Interactive

disagg_cluster:
  num_prefill_workers: 1
  num_decode_workers: 2
  system: GB200-NVL72_GB200-186GB_aarch64x36

prefill:
  system: GB200-NVL72_GB200-186GB_aarch64x4
  gpus_per_worker: 4
  gpus_per_node: 4
  config: /work/prefill_config.yaml

decode:
  system: GB200-NVL72_GB200-186GB_aarch64x16
  gpus_per_worker: 16
  gpus_per_node: 4
  config: /work/decode_config.yaml
```

### 2. Run the script `launch_disagg_cluster.sh`
```bash
# 1. Install dependencies (handy to use a venv for this)
pip install -r scripts/slurm_llm/dynamo_disagg/requirements.txt

# 2. Allocate nodes and launch
salloc --nodes=9 --time=2:00:00
python3 scripts/slurm_llm/dynamo_disagg/launch_disagg_cluster.py \
    --config scripts/slurm_llm/dynamo_disagg/config/sample/disagg_deployment_minimal.yaml \
    --container-image /path/to/image.sqsh
```

### 3. Optionally, `sbatch` your run
```bash
# Use the provided sbatch script:
sbatch --nodes=9 --time 2:0:0 -p backfill scripts/slurm_llm/dynamo_disagg/example_batch_script.sh
```

Or create your own:
```bash
#!/bin/bash

# Set up venv with pip
python3 -m venv --clear /tmp/disagg_venv
source /tmp/disagg_venv/bin/activate
python3 -m ensurepip --upgrade
pip install --upgrade pip
pip install -r scripts/slurm_llm/dynamo_disagg/requirements.txt

# Use -u for unbuffered output
python3 -u scripts/slurm_llm/dynamo_disagg/launch_disagg_cluster.py \
    --config scripts/slurm_llm/dynamo_disagg/config/sample/disagg_deployment_minimal.yaml \
    --container-image /path/to/image.sqsh
```

### 4. Check status
You should see three running job steps - one each for prefill workers, decode workers and frontend server.
```bash
$ sacct -j $SLURM_JOBID
JobID           JobName  Partition    Account  AllocCPUS      State ExitCode 
------------ ---------- ---------- ---------- ---------- ---------- -------- 
...
1522062.0          make            acct              144    RUNNING      0:0 
1522062.1          make            acct              144    RUNNING      0:0 
1522062.2          make            acct              576    RUNNING      0:0 
``` 

## Resource Allocation

`launch_disagg_cluster.py` automatically allocates nodes from your SLURM allocation to prefill and decode workers.

### Allocation Formula

Assuming `gpus_per_worker` is a multiple of `gpus_per_node`: 
  - TODO: Only inter-node for now. Intra-node is a WIP

```
nodes_per_prefill_worker = gpus_per_worker / gpus_per_node
nodes_per_decode_worker  = gpus_per_worker / gpus_per_node

prefill_nodes = num_prefill_workers × nodes_per_prefill_worker
decode_nodes  = num_decode_workers × nodes_per_decode_worker
total_nodes   = prefill_nodes + decode_nodes
```

The **frontend runs on the first prefill node** (no dedicated node).

### Node Assignment

Nodes from `SLURM_JOB_NODELIST` are assigned sequentially:

```
Prefill nodes:  [node0, node1, ...]        ← Frontend runs on node0
Decode nodes:   [nodeN, nodeN+1, ...]
```

### Prefill + Decode Isolation

Prefill and decode workers are allocated to **separate nodes** (no node overlap between prefill and decode).
  - This will change with intra-node


## Manual invokation: Standing up prefill/decode workers

### 1a. Launch Frontend

```bash
srun --overlap --nodes=1 --ntasks=1 -w $FRONTEND_NODE \
    --container-image=$CONTAINER_IMAGE \
    --container-mounts="$(pwd):/work" \
    --container-workdir=/work --container-remap-root \
    make run_llm_server RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Interactive \
        --core_type=disagg_frontend" &
```

### 1b. Launch Prefill Worker

```bash
srun --overlap --nodes=1 --ntasks=4 -w $PREFILL_NODE \
    --container-image=$CONTAINER_IMAGE \
    --container-mounts="$(pwd):/work" \
    --container-workdir=/work --container-remap-root \
    --mpi=pmix \
    make run_llm_server RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Interactive \
        --core_type=disagg_prefill \
        --dynamo_frontend_host=$FRONTEND_NODE \
        --trtllm_yml_override=/work/prefill_config.yaml \
        --mpi_mode=leader" &
```

### 1c. Launch Decode Worker

```bash
srun --overlap --nodes=4 --ntasks=16 -w $DECODE_NODES \
    --container-image=$CONTAINER_IMAGE \
    --container-mounts="$(pwd):/work" \
    --container-workdir=/work --container-remap-root \
    --mpi=pmix \
    make run_llm_server RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Interactive \
        --core_type=disagg_decode \
        --dynamo_frontend_host=$FRONTEND_NODE \
        --trtllm_yml_override=/work/decode_config.yaml \
        --mpi_mode=leader" &
```

### 1d. Run Benchmark

Wait for workers to register, then:

```bash
srun --overlap --nodes=1 --ntasks=1 -w $FRONTEND_NODE \
    --container-image=$CONTAINER_IMAGE \
    --container-mounts="$(pwd):/work" \
    --container-workdir=/work --container-remap-root \
    --export=MLPINF_HTTP_USE_COMPLETIONS=1 \
    make run_harness RUN_ARGS="--benchmarks=deepseek-r1 --scenarios=Interactive \
        --core_type=dynamo_endpoint \
        --trtllm_server_urls=$FRONTEND_NODE:8000 \
        --test_mode=AccuracyOnly \
        --server_target_qps=10"
```


---

## Checking in your configuration to `configs`

Internalize YAML configs into the MLPerf config system for reproducible deployments and sharing performant configs.

### Move YAML Configs

```bash
mkdir -p configs/GB200-NVL72_GB200-186GB_aarch64x4/Interactive/disagg_prefill
mkdir -p configs/GB200-NVL72_GB200-186GB_aarch64x16/Interactive/disagg_decode

mv prefill_config.yaml configs/GB200-NVL72_GB200-186GB_aarch64x4/Interactive/disagg_prefill/deepseek-r1.yml
mv decode_config.yaml configs/GB200-NVL72_GB200-186GB_aarch64x16/Interactive/disagg_decode/deepseek-r1.yml
```

### 3b. Create System Config

Create `configs/GB200-NVL72_GB200-186GB_aarch64x36/Interactive/deepseek-r1.py`:

```python
import code.llmlib.fields as llm_fields

dynamo_endpoint = {
    llm_fields.dynamo_cluster: {
        'num_prefill_workers': 1,
        'num_decode_workers': 2,
        'gpus_per_node': 4,
        'prefill': {
            'system': 'GB200-NVL72_GB200-186GB_aarch64x4',
            'trtllm_yml_override': '/work/configs/GB200-NVL72_GB200-186GB_aarch64x4/Interactive/disagg_prefill/deepseek-r1.yml',
        },
        'decode': {
            'system': 'GB200-NVL72_GB200-186GB_aarch64x16',
            'trtllm_yml_override': '/work/configs/GB200-NVL72_GB200-186GB_aarch64x16/Interactive/disagg_decode/deepseek-r1.yml',
        },
    },
    # ... other fields
}
```

### 3c. Launch via System Name
With this, the checked in configuration under `configs` will be picked up

```bash
python3 scripts/slurm_llm/dynamo_disagg/launch_disagg_cluster.py \
    --system GB200-NVL72_GB200-186GB_aarch64x36 \
    --benchmark deepseek-r1 \
    --scenario Interactive \
    --container-image /path/to/image.sqsh
```

---

## CLI Reference

### launch_disagg_cluster.py

| Argument | Description |
|----------|-------------|
| `--config` | YAML cluster config (Step 2) |
| `--system` | MLPerf system name (Step 3) |
| `--benchmark` | Benchmark (e.g., `deepseek-r1`) |
| `--scenario` | Scenario (default: `Interactive`) |
| `--container-image` | Container image path (required) |
| `--prefill-yml` | Override prefill YAML |
| `--decode-yml` | Override decode YAML |
| `--run-harness` | Run harness after deployment |
| `--server-target-qps` | Target QPS (default: 1.0) |
| `--test-mode` | `PerformanceOnly`, `AccuracyOnly`, `SubmissionRun` |
| `--dry-run` | Print commands without executing |

### Core Types for `make run_llm_serve`

| Core Type | Description |
|-----------|-------------|
| `disagg_frontend` | Frontend (NATS + etcd + router) |
| `disagg_prefill` | Prefill worker |
| `disagg_decode` | Decode worker |

### Core Types for `make run_harness`
| `dynamo_endpoint` | mimics `trtllm_endpoint` and skips config gen+logging |

---

## Logs

```
build/logs/disagg_{benchmark}_slurm-{jobid}_{timestamp}/
├── cluster_0/
│   ├── frontend/*.stdout
│   ├── prefill/*.stdout
│   ├── decode/*.stdout
│   └── harness/*.stdout  (if --run-harness)
```
