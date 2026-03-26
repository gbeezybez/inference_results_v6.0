# GPT-OSS-120B

## Getting Started

### Download Model and Dataset
Please refer to the reference implementation README on mlcommons/inference for instructions to download datasets and model ([here](https://github.com/mlcommons/inference/tree/master/language/gpt-oss-120b#model-and-dataset-download))

### Download and Prepare Data

The GPT-OSS-120B benchmark uses separate datasets for accuracy and performance runs:

`/work/build/data` is a symlink that points to `$MLPERF_SCRATCH_PATH/data`

```bash
# Create data directories
mkdir -p /work/build/data/gpt-oss/v4/acc
mkdir -p /work/build/data/gpt-oss/v4/perf

# Under v4/acc, place:
# - input_ids_padded.npy
# - input_lens.npy  
# - acc_eval_ref.parquet (reference data for evaluation)

# Under v4/perf, place
# - input_ids_padded.npy
# - input_lens.npy
```

Expected data layout:

```
build/data/gpt-oss/v4/
├── acc/
│   ├── input_ids_padded.npy    # Tokenized inputs (padded)
│   ├── input_lens.npy          # Actual input lengths
│   └── acc_eval_ref.parquet    # Ground truth for accuracy evaluation
└── perf/
    ├── input_ids_padded.npy
    └── input_lens.npy
```

**Dataset Statistics:**

| Dataset | Samples | Max ISL | Max OSL | Benchmarks |
|---------|---------|---------|---------|------------|
| Accuracy | 4,395 | 3,072 | 32,768 | AIME (240), GPQA (990), LiveCodeBench (3,165) |
| Performance | 6,396 | 15,330 | 10,240 | Synthetic |

### Configuration

The benchmark supports test_mode-aware configuration. The harness automatically selects the appropriate dataset and generation config based on `--test_mode`:

- `--test_mode=AccuracyOnly`: Uses accuracy dataset, max_output_len=32768
- `--test_mode=PerformanceOnly`: Uses performance dataset, max_output_len=10240

See `configs/<SYSTEM_NAME>/Offline/gpt-oss-120b.py` for system-specific configurations.

## Run

Please use `scaleout` to run: 

```bash
sbatch --nodes=1 -p gb200 \
    ./scaleout/run_scaleout.sh \
    --stage all \
    --atomic-system GB200-NVL72_GB200-186GB_aarch64x1 \
    --gpus-per-node 4 \
    --dp-multiplicity 4 \
    --container-image docker://... \
    --run-args "--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint"
```


### Interactively

```bash
# Enter the MLPerf container
make prebuild
```

```bash
export RUN_ARGS="--benchmarks=gpt-oss-120b --scenarios=Offline --core_type=trtllm_endpoint --test_mode=PerformanceOnly"
export SYSTEM_NAME=B200-SXM-180GBx8
make run_llm_server
make run_harness
```

### Debugging accuracy Issues

1. Verify correct dataset is loaded (accuracy vs performance)
2. Check `max_seq_len` supports full output length (3072 + 32768 for accuracy)
3. Ensure generation config matches test mode
