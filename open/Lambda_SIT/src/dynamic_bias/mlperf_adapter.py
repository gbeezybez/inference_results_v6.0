import torch
import time
import queue
from dataclasses import dataclass
from typing import Optional

import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                         ".."))
sys.path.insert(0, REPO_ROOT)

from tensorrt_llm.dynamic_bias import trtllm_integration

# gpt-oss-120b
NUM_EXPERTS = 128
TOPK = 4
NUM_LAYERS = 36
HIDDEN_SIZE = 2880

try:
    import tensorrt_llm
    from tensorrt_llm.runtime import ModelRunnerCpp
    HAS_TRTLLM = True
except ImportError:
    HAS_TRTLLM = False

try:
    import mlperf_loadgen as lg
    HAS_LOADGEN = True
except ImportError:
    HAS_LOADGEN = False


@dataclass
class HarnessConfig:
    engine_dir: str = ""
    tp_size: int = 8
    ep_size: int = 8
    pp_size: int = 1
    max_batch_size: int = 128
    max_input_len: int = 1024
    max_output_len: int = 1024

    scenario: str = "Server"
    target_qps: float = 15.0
    target_latency_ms: float = 100.0

    router_config: Optional[trtllm_integration.DynamicBiasConfig] = None
    quant_mode: str = "fp8"

    def __post_init__(self):
        if self.router_config is None:
            self.router_config = trtllm_integration.DynamicBiasConfig()


class DynamicBiasTRTLLMHarness:
    """
    MLPerf harness with dynamic bias routing for gpt-oss-120b.

    Flow per decode step:
        for layer in 36 layers:
            logits = router_projection(hidden)
            indices, weights = hook.intercept_moe_layer(layer, hidden, router_w)
            output = dispatch_and_combine(...)
        hook.on_step_complete()  # ONE EMA update + Layer 3/4 safety checks
    """

    def __init__(self, config: HarnessConfig):
        self.config = config
        self.hook = trtllm_integration.TRTLLMRouterHook(config.router_config)

        self._query_queue: queue.Queue = queue.Queue()
        self._total_tokens = 0
        self._total_queries = 0
        self._total_ttft_ms = 0.0
        self._start_time = 0.0

        self._engine = None
        if HAS_TRTLLM and config.engine_dir:
            self._init_engine()

    def _init_engine(self):
        print(f"Loading TRT-LLM engine: {self.config.engine_dir}")
        print(f"  TP={self.config.tp_size} EP={self.config.ep_size}")
        print(f"  Quant={self.config.quant_mode}")
        rc = self.config.router_config
        print(f"  DynBias: α={rc.alpha} α2={rc.alpha2} guardrail={rc.guardrail_threshold} "
              f"β={rc.ema_beta}")
        self._engine = ModelRunnerCpp.from_dir(
            engine_dir=self.config.engine_dir, rank=0,
        )
        print("Engine loaded")

    def warmup(self, num_warmup: int = 10):
        """Warmup + alpha calibration."""
        print(f"Warmup ({num_warmup} iters)...")

        for i in range(num_warmup):
            num_tokens = self.config.max_batch_size * 32
            fake_hidden = torch.randn(num_tokens, HIDDEN_SIZE, device="cuda")
            fake_router_w = torch.randn(HIDDEN_SIZE, NUM_EXPERTS, device="cuda")

            for layer_idx in range(NUM_LAYERS):
                self.hook.intercept_moe_layer(
                    layer_idx, fake_hidden, fake_router_w,
                )
            self.hook.on_step_complete()

        # Print calibration
        print("Warmup done. Router stats:")
        for i in [0, 17, 35]:
            stats = self.hook.manager.router.get_layer_stats(i)
            print(f"  L{i:02d}: CV={stats['coeff_var']:.3f}, "
                  f"α={self.hook.manager.router.alpha:.3f}")

    def issue_queries(self, query_samples: list):
        for sample in query_samples:
            self._query_queue.put({
                "id": sample.id,
                "index": sample.index,
                "enqueue_time": time.perf_counter(),
            })

    def flush_queries(self):
        pass

    def process_batch(self):
        batch = []
        try:
            while len(batch) < self.config.max_batch_size:
                item = self._query_queue.get_nowait()
                batch.append(item)
        except queue.Empty:
            pass

        if not batch:
            return

        t_start = time.perf_counter()
        num_tokens = len(batch) * 32

        fake_hidden = torch.randn(num_tokens, HIDDEN_SIZE, device="cuda")
        fake_router_w = torch.randn(HIDDEN_SIZE, NUM_EXPERTS, device="cuda")

        for layer_idx in range(NUM_LAYERS):
            self.hook.intercept_moe_layer(
                layer_idx, fake_hidden, fake_router_w,
            )
        self.hook.on_step_complete()

        t_end = time.perf_counter()
        ttft_ms = (t_end - t_start) * 1000

        self._total_queries += len(batch)
        self._total_tokens += num_tokens
        self._total_ttft_ms += ttft_ms

        if HAS_LOADGEN:
            for item in batch:
                response = lg.QuerySampleResponse(item["id"], 0, 0)
                lg.QuerySamplesComplete([response])

    def get_perf_stats(self) -> dict:
        elapsed = time.perf_counter() - self._start_time if self._start_time else 0
        return {
            "total_queries": self._total_queries,
            "total_tokens": self._total_tokens,
            "avg_ttft_ms": self._total_ttft_ms / max(self._total_queries, 1),
            "qps": self._total_queries / max(elapsed, 1e-6),
            "tokens_per_sec": self._total_tokens / max(elapsed, 1e-6),
        }


def main():
    print("=== MLPerf Harness v3 Test ===\n")

    config = HarnessConfig(
        router_config=trtllm_integration.DynamicBiasConfig(
            alpha=0.15,
            alpha2=0.02,
            guardrail_threshold=1.5,
            ema_beta=0.85,
            alpha_auto_tune=True,
            alpha_safety_enabled=True,
            flip_monitor_enabled=True,
            target_cv=0.35,
        ),
    )

    harness = DynamicBiasTRTLLMHarness(config)
    harness.warmup(5)

    print("\nSimulated inference...")
    harness._start_time = time.perf_counter()

    for i in range(20):
        for _ in range(16):
            harness._query_queue.put({
                "id": i, "index": i,
                "enqueue_time": time.perf_counter(),
            })
        harness.process_batch()

    stats = harness.get_perf_stats()
    print(f"\nPerf: queries={stats['total_queries']} "
          f"avg_ttft={stats['avg_ttft_ms']:.2f}ms "
          f"qps={stats['qps']:.1f}")


if __name__ == "__main__":
    main()
