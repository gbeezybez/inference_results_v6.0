# Text-to-Video


Build the Docker image:
```
docker build -f Dockerfile -t mlperf-inference-xdit .
```



Launch a container:
```
docker run \
    -it \
    --rm \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ipc=host \
    --network host \
    --privileged \
    --shm-size 128G \
    --name ${USER}-mlperf-inference-xdit \
    -v ./outputs:/app/mlperf/mlperf_inference/text_to_video/wan-2.2-t2v-a14b/outputs \
    -w /app/mlperf/mlperf_inference/text_to_video/wan-2.2-t2v-a14b/ \
    mlperf-inference-xdit:latest \
    /bin/bash
```

If you prefer you can also modify the `docker run` command above to map your `HF_HOME` directory to skip downloading the model:
```
-v /your/hf/home/on/host:/hf_cache \
-e HF_HOME="/hf_cache" \
```

If you need to download the model, run inside the container:
```
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B-Diffusers
```

# Running scenarios

Use `run_scenarios.sh` to run accuracy, (optional) VBench, performance, and compliance in one go:

```bash
# Run both SingleStream and Offline (default)
./run_scenarios.sh --sage-fraction 0.85
```


Optional:
```bash
# Run only one scenario
./run_scenarios.sh SingleStream
./run_scenarios.sh Offline

# Skip VBench or compliance
./run_scenarios.sh --skip-vbench
./run_scenarios.sh --skip-compliance

# Preview commands without executing
./run_scenarios.sh --dry-run
./run_scenarios.sh --help
```

Script expects to be run from `/app/mlperf/mlperf_inference/text_to_video/wan-2.2-t2v-a14b/` (e.g. inside the Docker container). Set `MLPERF_ROOT` if your mlperf repo root is elsewhere.
