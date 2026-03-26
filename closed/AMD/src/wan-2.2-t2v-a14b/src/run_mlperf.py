
import argparse
import yaml
import json
import logging
import os
import torch
import array
import numpy as np
import mlperf_loadgen as lg
from pathlib import Path
from model import Model, ModelConfig


SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}

NANO_SEC = 1e9
MILLI_SEC = 1000


def setup_logging(rank):
    """Setup logging configuration for data parallel (all ranks log)."""
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_prompts(dataset_path):
    """Load prompts from dataset file."""
    with open(dataset_path, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def load_query_samples(sample_list):
    pass


def unload_query_samples(sample_list):
    pass


def get_args():
    parser = argparse.ArgumentParser(
        description="Batch T2V inference with Wan2.2-Diffusers")
    # Model Arguments
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/Wan2.2-T2V-A14B-Diffusers",
        help="Path to model checkpoint directory (default: ./models/Wan2.2-T2V-A14B-Diffusers)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/vbench_prompts.txt",
        help="Path to dataset file (text prompts, one per line) (default: ./data/prompts.txt)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save generated videos (default: ./data/outputs)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./inference_config.yaml",
        help="Path to inference configuration file (default: ./inference_config.yaml)"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1,
        help="Number of generation iterations per prompt (default: 1)"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=-1,
        help="Process only first N prompts (for testing, default: all)"
    )
    parser.add_argument(
        "--fixed-latent",
        type=str,
        default="./data/fixed_latent.pt",
        help="Path to fixed latent .pt file for deterministic generation (default: data/fixed_latent.pt)"
    )
    # MLPerf loadgen arguments
    parser.add_argument(
        "--scenario",
        default="SingleStream",
        help="mlperf benchmark scenario, one of " +
        str(list(SCENARIO_MAP.keys())),
    )
    parser.add_argument(
        "--user_conf",
        default="user.conf",
        help="user config for user LoadGen settings such as target QPS",
    )
    parser.add_argument(
        "--audit_conf", default="audit.config", help="config for LoadGen audit settings"
    )
    parser.add_argument(
        "--performance-sample-count",
        type=int,
        help="performance sample count",
        default=248,
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="enable accuracy pass"
    )
    parser.add_argument(
        "--video_output_path",
        type=str,
        default=None,
        help=f"path to store output videos, defaults to args.output_dir/videos"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help=f"target batch size if multiple samples are requested"
    )
    # Dont overwrite these for official submission
    parser.add_argument("--count", type=int, help="dataset items to use")
    parser.add_argument("--time", type=int, help="time to scan in seconds")
    parser.add_argument("--qps", type=int, help="target qps")
    parser.add_argument("--debug", action="store_true", help="debug")
    parser.add_argument(
        "--samples-per-query",
        default=8,
        type=int,
        help="mlperf multi-stream samples per query",
    )
    parser.add_argument(
        "--max-latency", type=float, help="mlperf max latency in pct tile"
    )
    parser.add_argument(
        "--sage-fraction",
        type=float,
        default=0.9,
        help="Fraction of steps using SAGE attention (default: 0.9)",
    )

    return parser.parse_args()

class DebugModel:
    def __init__(
        self, model_path, device, config, prompts, fixed_latent=None, rank=0
    ):
        self.prompts = prompts

    def issue_queries(self, query_samples):
        idx = [q.index for q in query_samples]
        query_ids = [q.id for q in query_samples]
        logging.info(f"Received query with {len(query_samples)} samples")
        response = []
        for i, q in zip(idx, query_ids):
            # print(i, self.prompts[i])
            output = self.prompts[i]
            response_array = array.array(
                "B", output.encode("utf-8")
            )
            bi = response_array.buffer_info()
            response.append(lg.QuerySampleResponse(q, bi[0], bi[1]))
        lg.QuerySamplesComplete(response)

    def flush_queries(self):
        pass

def run_mlperf(args, config):
    # Load dataset
    dataset = load_prompts(args.dataset)

    # Load model parameters
    # Parallelism parameters
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    setup_logging(rank)

    # Generation parameters from config

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_lg = str(args.output_dir)

    if args.video_output_path is None:
        video_output_path = Path(output_dir / "videos")
    else:
        video_output_path = Path(args.video_output_path)
    video_output_path.mkdir(parents=True, exist_ok=True)


    fixed_latent = None
    if args.fixed_latent:
        fixed_latent = torch.load(args.fixed_latent)
        logging.info(
            f"Loaded fixed latent from {args.fixed_latent} with shape: {fixed_latent.shape}")
        logging.info(f"This latent will be reused for all generations")
    else:
        logging.info("No fixed latent provided - using random initial latents")

    # Loading model
    model_config = ModelConfig(
        negative_prompt=config['negative_prompt'].strip(),
        guidance_scale=config['guidance_scale'],
        guidance_scale_2=config['guidance_scale_2'],
        num_frames=config['num_frames'],
        num_inference_steps=config['sample_steps'],
        height=config['height'],
        width=config['width'],
        base_seed=config['seed'],
        fps=config['fps'],
        sage_fraction=args.sage_fraction,
    )
    model = Model(
        model_path=args.model_path, 
        prompts=dataset, 
        config=model_config, 
        fixed_latent=fixed_latent, 
        batch_size=args.batch_size, 
        video_output_path=video_output_path,
        performance_sample_count=args.performance_sample_count
    )
    # model = DebugModel(model_path=args.model_path, device=device, config=config, prompts=dataset, fixed_latent=fixed_latent, rank=rank)
    logging.info("Model loaded successfully!")

    # Prepare loadgen for run
    if rank == 0:
        # Rank 0 runs LoadGen
        log_output_settings = lg.LogOutputSettings()
        log_output_settings.outdir = output_dir_lg
        log_output_settings.prefix = "mlperf_log_"
        log_output_settings.copy_summary_to_stdout = False

        log_settings = lg.LogSettings()
        log_settings.enable_trace = args.debug
        log_settings.log_output = log_output_settings

        user_conf = os.path.abspath(args.user_conf)
        settings = lg.TestSettings()
        settings.FromConfig(user_conf, "wan-2.2-t2v-a14b", args.scenario)

        audit_config = os.path.abspath(args.audit_conf)
        if os.path.exists(audit_config):
            settings.FromConfig(audit_config, "wan-2.2-t2v-a14b", args.scenario)
        settings.scenario = SCENARIO_MAP[args.scenario]

        settings.mode = lg.TestMode.PerformanceOnly
        if args.accuracy:
            settings.mode = lg.TestMode.AccuracyOnly

        if args.time:
            # override the time we want to run
            settings.min_duration_ms = args.time * MILLI_SEC
            settings.max_duration_ms = args.time * MILLI_SEC
        if args.qps:
            qps = float(args.qps)
            settings.server_target_qps = qps
            settings.offline_expected_qps = qps

        count_override = False
        count = args.count
        if count:
            count_override = True

        if args.count:
            settings.min_query_count = count
            settings.max_query_count = count
        if not count_override:
            count = len(dataset)

        if args.samples_per_query:
            settings.multi_stream_samples_per_query = args.samples_per_query
        if args.max_latency:
            settings.server_target_latency_ns = int(
                args.max_latency * NANO_SEC)
            settings.multi_stream_expected_latency_ns = int(
                args.max_latency * NANO_SEC)

        performance_sample_count = (
            args.performance_sample_count
            if args.performance_sample_count
            else min(count, 500)
        )

        sut = lg.ConstructSUT(model.issue_queries, model.flush_queries)
        qsl = lg.ConstructQSL(
            count, performance_sample_count, load_query_samples, unload_query_samples
        )

        logging.info("Starting LoadGen test on rank 0...")
        lg.StartTestWithLogSettings(
            sut, qsl, settings, log_settings, audit_config)
        
        if args.accuracy:
            # TODO: output accuracy
            final_results = {}
            with open("results.json", "w") as f:
                json.dump(final_results, f, sort_keys=True, indent=4)

        lg.DestroyQSL(qsl)
        lg.DestroySUT(sut)
        
        # Save videos to filesystem if requested
        logging.info("Saving videos to filesystem...")
        model.save_videos_to_filesystem()
        
        # Signal worker ranks that we're done
        logging.info("LoadGen test complete, signaling workers...")
        model.signal_workers_done()
        
        # Cleanup resources
        model.cleanup()
        logging.info("All workers signaled, rank 0 exiting")
    else:
        # Non-zero ranks enter worker loop to participate in distributed inference
        logging.info(f"Rank {rank} entering worker loop to participate in distributed inference")
        model.worker_loop()
        logging.info(f"Rank {rank} exiting worker loop")
        
        # Cleanup resources
        model.cleanup()


def main():
    args = get_args()
    config = load_config(args.config)
    run_mlperf(args, config)


if __name__ == "__main__":
    main()
