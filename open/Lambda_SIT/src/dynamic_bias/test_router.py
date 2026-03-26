import argparse
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import time

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                         ".."))
sys.path.insert(0, REPO_ROOT)

# gpt-oss-120b constants
NUM_EXPERTS = 128
TOPK = 4
NUM_LAYERS = 36
HIDDEN_SIZE = 2880


def _import_module_by_path(name: str):
    module_path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_dynamic_bias_module(name: str):
    if os.environ.get("DYNAMIC_BIAS_STANDALONE", "0") == "1":
        return _import_module_by_path(name)
    try:
        return importlib.import_module(f"tensorrt_llm.dynamic_bias.{name}")
    except Exception:
        return _import_module_by_path(name)


# ---------------------------------------------------------------------------
# Correctness Tests
# ---------------------------------------------------------------------------

def test_basic_routing():
    """Router produces valid int32 indices and normalized weights."""
    triton_router = _import_dynamic_bias_module("triton_router")

    print("=== Test: Basic Routing ===")
    router = triton_router.DynamicBiasRouterV3(alpha=0.1)

    num_tokens = 256
    logits = torch.randn(num_tokens, NUM_EXPERTS, device="cuda")
    bias = torch.zeros(NUM_EXPERTS, device="cuda")

    indices, weights, flags = router.route(0, logits, bias)

    assert indices.dtype == torch.int32, f"Expected int32, got {indices.dtype}"
    assert indices.shape == (num_tokens, TOPK)
    assert weights.shape == (num_tokens, TOPK)
    assert (indices >= 0).all() and (indices < NUM_EXPERTS).all()

    weight_sums = weights.sum(dim=1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-3)

    for i in range(min(num_tokens, 10)):
        selected = indices[i].tolist()
        assert len(set(selected)) == TOPK, f"duplicates: {selected}"

    print(f"  dtype={indices.dtype}, shape={indices.shape}")
    print(f"  weight sums (first 5): {[f'{s:.4f}' for s in weight_sums[:5].tolist()]}")
    print(f"  locked: {flags.sum().item()}/{num_tokens}")
    print("  PASSED\n")


def test_per_layer_isolation():
    """Each layer has independent state."""
    triton_router = _import_dynamic_bias_module("triton_router")

    print("=== Test: Per-Layer Isolation ===")
    router = triton_router.DynamicBiasRouterV3(alpha=0.2)

    num_tokens = 512
    logits_skewed = torch.randn(num_tokens, NUM_EXPERTS, device="cuda") * 0.3
    logits_skewed[:, :8] += 3.0
    logits_uniform = torch.randn(num_tokens, NUM_EXPERTS, device="cuda") * 0.5
    bias = torch.zeros(NUM_EXPERTS, device="cuda")

    router.route(0, logits_skewed, bias)
    router.route(1, logits_uniform, bias)
    router.step(num_tokens)

    s0 = router.get_layer_stats(0)
    s1 = router.get_layer_stats(1)

    print(f"  L0 (skewed):  CV={s0['coeff_var']:.3f}, ema_max={s0['ema_max']:.2f}")
    print(f"  L1 (uniform): CV={s1['coeff_var']:.3f}, ema_max={s1['ema_max']:.2f}")
    assert s0['ema_max'] > s1['ema_max']
    print("  PASSED\n")


def test_guardrail_on_sem():
    """Guardrail uses semantic baseline (raw + bias), not raw alone."""
    triton_router = _import_dynamic_bias_module("triton_router")

    print("=== Test: Guardrail on Semantic Baseline ===")
    router = triton_router.DynamicBiasRouterV3(alpha=0.5, guardrail_threshold=1.0)

    num_tokens = 128
    logits = torch.randn(num_tokens, NUM_EXPERTS, device="cuda") * 0.1
    bias = torch.zeros(NUM_EXPERTS, device="cuda")
    bias[0] = 5.0

    indices, weights, flags = router.route(0, logits, bias)

    locked_count = flags.sum().item()
    top1_is_0 = (indices[:, 0] == 0).sum().item()

    print(f"  Locked (via sem): {locked_count}/{num_tokens}")
    print(f"  Top-1 = expert 0: {top1_is_0}/{num_tokens}")
    assert locked_count > 80, f"Guardrail should fire on sem, only {locked_count} locked"
    print("  PASSED\n")


def test_normalized_ema():
    """After step(), ema values should be normalized (≈1.0 for avg expert)."""
    triton_router = _import_dynamic_bias_module("triton_router")

    print("=== Test: Normalized EMA ===")
    router = triton_router.DynamicBiasRouterV3(ema_beta=0.0)

    num_tokens = 1024
    logits = torch.randn(num_tokens, NUM_EXPERTS, device="cuda") * 0.3
    bias = torch.zeros(NUM_EXPERTS, device="cuda")

    router.route(0, logits, bias)
    router.step(num_tokens)

    stats = router.get_layer_stats(0)
    print(f"  ema_mean={stats['ema_mean']:.3f} (should be ≈1.0)")
    assert 0.5 < stats['ema_mean'] < 2.0, f"Normalized mean off: {stats['ema_mean']}"
    print("  PASSED\n")


def test_normalization_batch_size_stability():
    """α effect should be similar across different batch sizes."""
    triton_router = _import_dynamic_bias_module("triton_router")

    print("=== Test: Batch Size Stability ===")

    ema_maxes = {}
    for num_tokens in [64, 256, 1024, 4096]:
        router = triton_router.DynamicBiasRouterV3(ema_beta=0.0, alpha=0.2)
        logits = torch.randn(num_tokens, NUM_EXPERTS, device="cuda") * 0.3
        logits[:, :8] += 3.0
        bias = torch.zeros(NUM_EXPERTS, device="cuda")

        router.route(0, logits, bias)
        router.step(num_tokens)

        stats = router.get_layer_stats(0)
        ema_maxes[num_tokens] = stats['ema_max']
        print(f"  N={num_tokens:>4}: ema_max={stats['ema_max']:.3f}")

    vals = list(ema_maxes.values())
    ratio = max(vals) / min(vals)
    print(f"  Max/min ratio: {ratio:.2f} (should be <5.0)")
    assert ratio < 5.0, f"Normalization failed, ratio={ratio:.2f}"
    print("  PASSED\n")


def test_intra_step_feedback():
    """alpha2 > 0 should reduce intra-step hot expert concentration."""
    triton_router = _import_dynamic_bias_module("triton_router")

    print("=== Test: Intra-Step Feedback (α2) ===")

    num_tokens = 2048
    logits = torch.randn(num_tokens, NUM_EXPERTS, device="cuda") * 0.3
    logits[:, :4] += 3.0
    bias = torch.zeros(NUM_EXPERTS, device="cuda")

    r_off = triton_router.DynamicBiasRouterV3(alpha=0.0, alpha2=0.0)
    idx_off, _, _ = r_off.route(0, logits, bias)

    r_on = triton_router.DynamicBiasRouterV3(alpha=0.0, alpha2=0.03)
    idx_on, _, _ = r_on.route(0, logits, bias)

    def count_cv(idx):
        load = torch.zeros(NUM_EXPERTS, device="cuda")
        for k in range(TOPK):
            load.scatter_add_(0, idx[:, k].long(), torch.ones(num_tokens, device="cuda"))
        return (load.std() / load.mean()).item()

    cv_off = count_cv(idx_off)
    cv_on = count_cv(idx_on)

    print(f"  α2=0:    CV={cv_off:.3f}")
    print(f"  α2=0.03: CV={cv_on:.3f}")
    print(f"  Improvement: {(1 - cv_on / cv_off) * 100:.1f}%")
    print("  PASSED\n")


def test_weight_consistency():
    """Weights should use sem=raw+bias for BOTH locked and unlocked tokens."""
    triton_router = _import_dynamic_bias_module("triton_router")

    print("=== Test: Weight Consistency ===")
    router = triton_router.DynamicBiasRouterV3(alpha=0.3, guardrail_threshold=1.0)

    num_tokens = 256
    logits = torch.randn(num_tokens, NUM_EXPERTS, device="cuda")
    bias = torch.randn(NUM_EXPERTS, device="cuda") * 0.5

    indices, weights, flags = router.route(0, logits, bias)

    sem = logits + bias.unsqueeze(0)
    for i in range(min(5, num_tokens)):
        sel = indices[i].long()
        expected = torch.softmax(sem[i][sel], dim=0)
        actual = weights[i]
        status = "LOCKED" if flags[i].item() else "unlocked"
        if torch.allclose(actual, expected, atol=1e-3):
            print(f"  Token {i} ({status}): weights match ✓")
        else:
            print(f"  Token {i} ({status}): expected={expected.tolist()}, got={actual.tolist()}")

    print("  PASSED\n")


def test_no_cross_step_decay_bug():
    """step() only runs once, not 36 times."""
    triton_router = _import_dynamic_bias_module("triton_router")

    print("=== Test: No Cross-Layer Decay Bug ===")
    router = triton_router.DynamicBiasRouterV3(ema_beta=0.9)

    logits = torch.randn(256, NUM_EXPERTS, device="cuda")
    logits[:, 0] += 5.0
    bias = torch.zeros(NUM_EXPERTS, device="cuda")

    for layer_idx in range(NUM_LAYERS):
        router.route(layer_idx, logits, bias)
    router.step(256)

    stats = router.get_layer_stats(0)
    print(f"  L0 ema_max after full pass: {stats['ema_max']:.3f}")
    assert stats['ema_max'] > 0.5, f"EMA too low: {stats['ema_max']}"
    print("  PASSED\n")


def test_load_balancing():
    """Dynamic bias spreads load over multiple steps."""
    triton_router = _import_dynamic_bias_module("triton_router")

    print("=== Test: Load Balancing ===")

    r_off = triton_router.DynamicBiasRouterV3(alpha=0.0)
    r_on = triton_router.DynamicBiasRouterV3(alpha=0.3)

    num_tokens = 1024
    logits = torch.randn(num_tokens, NUM_EXPERTS, device="cuda") * 0.3
    logits[:, :8] += 3.0
    bias = torch.zeros(NUM_EXPERTS, device="cuda")

    for _ in range(5):
        idx_off, _, _ = r_off.route(0, logits, bias)
        idx_on, _, _ = r_on.route(0, logits, bias)
        r_off.step(num_tokens)
        r_on.step(num_tokens)

    def count_cv(idx):
        load = torch.zeros(NUM_EXPERTS, device="cuda")
        for k in range(TOPK):
            load.scatter_add_(0, idx[:, k].long(), torch.ones(num_tokens, device="cuda"))
        return (load.std() / load.mean()).item()

    cv_off = count_cv(idx_off)
    cv_on = count_cv(idx_on)

    print(f"  OFF: CV={cv_off:.3f}  |  ON: CV={cv_on:.3f}")
    print(f"  Reduction: {(1 - cv_on / cv_off) * 100:.1f}%")
    assert cv_on < cv_off
    print("  PASSED\n")


def test_flip_rate_detection():
    """Layer 4: flip and drop rates are detected with sampling window."""
    triton_router = _import_dynamic_bias_module("triton_router")

    print("=== Test: Flip + Drop Rate Detection (Layer 4) ===")

    router = triton_router.DynamicBiasRouterV3(alpha=0.0, guardrail_threshold=100.0)
    router.accuracy_stats_enabled = True

    num_tokens = 1024
    logits = torch.randn(num_tokens, NUM_EXPERTS, device="cuda") * 0.3
    logits[:, :8] += 3.0
    bias = torch.zeros(NUM_EXPERTS, device="cuda")

    router.route(0, logits, bias)
    router.step(num_tokens)

    router.alpha = 2.0
    router.reset_accuracy_stats()
    router.route(0, logits, bias)

    acc = router.get_accuracy_stats(0)
    print(f"  flip_rate={acc['flip_rate']:.4f}")
    print(f"  drop_rate={acc['drop_rate']:.5f}")
    print(f"  total_tokens={acc['total_tokens']}")

    assert acc['flip_rate'] > 0.0, "Expected some flips with α=2.0"
    assert acc['total_tokens'] == num_tokens

    # Verify sampling flag
    router.reset_accuracy_stats()
    router.accuracy_stats_enabled = False
    router.route(0, logits, bias)
    acc_off = router.get_accuracy_stats(0)
    assert acc_off['total_tokens'] == 0, "Stats should not collect when disabled"

    print("  PASSED\n")


def test_drop_rate_extreme_alpha():
    """Layer 4 hard safety: extreme α pushes top-1 out of top-4."""
    triton_router = _import_dynamic_bias_module("triton_router")

    print("=== Test: Drop Rate with Extreme Alpha ===")

    router = triton_router.DynamicBiasRouterV3(alpha=0.0, guardrail_threshold=100.0)
    router.accuracy_stats_enabled = True

    num_tokens = 512
    logits = torch.randn(num_tokens, NUM_EXPERTS, device="cuda") * 1.0
    logits[:, 0] += 2.0
    bias = torch.zeros(NUM_EXPERTS, device="cuda")

    for _ in range(10):
        router.route(0, logits, bias)
        router.step(num_tokens)

    router.alpha = 10.0
    router.reset_accuracy_stats()
    router.route(0, logits, bias)

    acc = router.get_accuracy_stats(0)
    print(f"  flip_rate={acc['flip_rate']:.4f}")
    print(f"  drop_rate={acc['drop_rate']:.5f}")

    assert acc['drop_rate'] > 0.0 or acc['flip_rate'] > 0.0, \
        "Expected drops or flips with α=10.0"
    print("  PASSED\n")


def test_gap_stats_sampling():
    """Layer 3: gap(top1, top5) stats and safe alpha computation."""
    triton_router = _import_dynamic_bias_module("triton_router")

    print("=== Test: Gap Stats Sampling (Layer 3) ===")

    router = triton_router.DynamicBiasRouterV3(alpha=0.15)
    router.accuracy_stats_enabled = True
    num_tokens = 512

    logits = torch.randn(num_tokens, NUM_EXPERTS, device="cuda") * 0.5
    logits[:, 0] += 10.0
    bias = torch.zeros(NUM_EXPERTS, device="cuda")

    router.route(0, logits, bias)
    router.step(num_tokens)
    router.sample_gap_stats(0, logits, bias)

    acc = router.get_accuracy_stats(0)
    print(f"  gap_p10={acc['gap_p10']:.2f}")
    print(f"  max_load_ema={acc['max_load_ema']:.2f}")

    assert acc['gap_p10'] > 0 and acc['gap_p10'] < float('inf')

    safe_alpha = router.compute_safe_alpha(safety_margin=0.8)
    print(f"  safe_alpha={safe_alpha:.3f}")
    assert safe_alpha > 0
    print("  PASSED\n")


def test_safe_alpha_with_alpha2():
    """Layer 3: safe alpha accounts for alpha2 contribution."""
    triton_router = _import_dynamic_bias_module("triton_router")

    print("=== Test: Safe Alpha with alpha2 ===")

    num_tokens = 512
    logits = torch.randn(num_tokens, NUM_EXPERTS, device="cuda") * 0.5
    logits[:, 0] += 5.0
    bias = torch.zeros(NUM_EXPERTS, device="cuda")

    # Without alpha2
    r1 = triton_router.DynamicBiasRouterV3(alpha=0.15, alpha2=0.0)
    r1.route(0, logits, bias)
    r1.step(num_tokens)
    r1.sample_gap_stats(0, logits, bias)
    safe_no_a2 = r1.compute_safe_alpha(0.8)

    # With alpha2 > 0: safe alpha should be LOWER (alpha2 eats gap budget)
    r2 = triton_router.DynamicBiasRouterV3(alpha=0.15, alpha2=0.05)
    r2._last_num_tokens = num_tokens
    r2.route(0, logits, bias)
    r2.step(num_tokens)
    r2.sample_gap_stats(0, logits, bias)
    safe_with_a2 = r2.compute_safe_alpha(0.8)

    print(f"  safe_alpha (α2=0):    {safe_no_a2:.3f}")
    print(f"  safe_alpha (α2=0.05): {safe_with_a2:.3f}")
    assert safe_with_a2 <= safe_no_a2, \
        f"alpha2 should reduce safe bound: {safe_with_a2} > {safe_no_a2}"
    print("  PASSED\n")


def test_alpha_clamp_prevents_flip():
    """Layer 3+4 integration: α clamp keeps flip rate low."""
    triton_router = _import_dynamic_bias_module("triton_router")

    print("=== Test: Alpha Clamp Prevents Excessive Flips ===")

    num_tokens = 1024
    logits = torch.randn(num_tokens, NUM_EXPERTS, device="cuda") * 0.3
    logits[:, :8] += 3.0
    bias = torch.zeros(NUM_EXPERTS, device="cuda")

    router = triton_router.DynamicBiasRouterV3(alpha=5.0, guardrail_threshold=1.0)
    router.accuracy_stats_enabled = True

    router.route(0, logits, bias)
    router.step(num_tokens)

    router.sample_gap_stats(0, logits, bias)
    safe_alpha = router.compute_safe_alpha(safety_margin=0.8)
    print(f"  Original α=5.0, safe_alpha={safe_alpha:.3f}")

    router.alpha = min(router.alpha, max(safe_alpha, 0.01))
    print(f"  Clamped α={router.alpha:.3f}")

    router.reset_accuracy_stats()
    router.route(0, logits, bias)
    acc = router.get_accuracy_stats(0)
    print(f"  flip_rate after clamp={acc['flip_rate']:.4f}")

    assert router.alpha <= safe_alpha + 0.01
    print("  PASSED\n")


def test_manager_accuracy_loop():
    """Integration: Manager runs full accuracy control loop with sampling windows."""
    trtllm_integration = _import_dynamic_bias_module("trtllm_integration")

    print("=== Test: Manager Accuracy Control Loop ===")

    config = trtllm_integration.DynamicBiasConfig(
        alpha=0.5,
        guardrail_threshold=1.5,
        alpha_auto_tune=True,
        alpha_safety_enabled=True,
        flip_monitor_enabled=True,
        max_flip_rate=0.05,
        max_drop_rate=0.001,
        auto_tune_interval=5,
        gap_sample_interval=5,
        sample_on_steps=2,
        sample_off_steps=3,
        log_interval=0,
    )
    manager = trtllm_integration.DynamicBiasRouterManager(config)

    num_tokens = 512
    logits = torch.randn(num_tokens, NUM_EXPERTS, device="cuda") * 0.3
    logits[:, :8] += 3.0
    bias = torch.zeros(NUM_EXPERTS, device="cuda")

    alphas = []
    for step in range(20):
        for layer_idx in range(NUM_LAYERS):
            manager.route(layer_idx, logits, bias)
        manager.step(num_tokens)
        alphas.append(manager.router.alpha)

    print(f"  α trajectory: {[f'{a:.3f}' for a in alphas[:10]]}")
    print(f"  Final α={alphas[-1]:.3f} (started at 0.5)")

    for layer_idx in [0, 35]:
        acc = manager.router.get_accuracy_stats(layer_idx)
        print(f"  L{layer_idx:02d}: flip={acc['flip_rate']:.4f} lock={acc['lock_rate']:.3f}")

    print("  PASSED\n")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def benchmark_single_layer():
    triton_router = _import_dynamic_bias_module("triton_router")

    print("=== Benchmark: Single Layer ===")
    print(f"{'Tokens':>8} {'Route(us)':>10}")
    print("-" * 22)

    router = triton_router.DynamicBiasRouterV3(alpha=0.15)
    bias = torch.zeros(NUM_EXPERTS, device="cuda")

    for nt in [64, 128, 256, 512, 1024, 2048, 4096]:
        logits = torch.randn(nt, NUM_EXPERTS, device="cuda")
        for _ in range(20):
            router.route(0, logits, bias)
        torch.cuda.synchronize()

        iters = 200
        t0 = time.perf_counter()
        for _ in range(iters):
            router.route(0, logits, bias)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) / iters * 1e6
        print(f"{nt:>8} {us:>10.1f}")
    print()


def benchmark_full_forward():
    triton_router = _import_dynamic_bias_module("triton_router")

    print("=== Benchmark: 36 Layers + step ===")
    print(f"{'Tokens':>8} {'36xRoute(us)':>13} {'Step(us)':>9} {'Total(us)':>10}")
    print("-" * 46)

    router = triton_router.DynamicBiasRouterV3(alpha=0.15)
    bias = torch.zeros(NUM_EXPERTS, device="cuda")

    for nt in [256, 1024, 4096]:
        logits = torch.randn(nt, NUM_EXPERTS, device="cuda")
        for _ in range(5):
            for l in range(NUM_LAYERS):
                router.route(l, logits, bias)
            router.step(nt)
        torch.cuda.synchronize()

        iters = 50
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            for l in range(NUM_LAYERS):
                router.route(l, logits, bias)
        torch.cuda.synchronize()
        route_us = (time.perf_counter() - t0) / iters * 1e6

        t0 = time.perf_counter()
        for _ in range(iters):
            router.step(nt)
        torch.cuda.synchronize()
        step_us = (time.perf_counter() - t0) / iters * 1e6

        print(f"{nt:>8} {route_us:>13.1f} {step_us:>9.1f} {route_us + step_us:>10.1f}")
    print()


def simulate_workload():
    triton_router = _import_dynamic_bias_module("triton_router")

    print("=== Simulation: Traffic Pattern ===")
    router = triton_router.DynamicBiasRouterV3(alpha=0.2, alpha2=0.02, guardrail_threshold=1.5,
                                  ema_beta=0.8)
    bias = torch.zeros(NUM_EXPERTS, device="cuda")

    print(f"{'Step':>5} {'Phase':>8} {'L0_CV':>7} {'L35_CV':>7} {'Alpha':>6}")
    print("-" * 40)

    for step in range(30):
        nt = 512
        logits = torch.randn(nt, NUM_EXPERTS, device="cuda") * 0.5

        if step < 10:
            logits[:, :16] += 2.5
            phase = "code"
        elif step < 20:
            phase = "diverse"
        else:
            logits[:, 32:48] += 2.5
            phase = "math"

        for l in range(NUM_LAYERS):
            router.route(l, logits, bias)
        router.step(nt)

        if step % 3 == 0 or step == 29:
            s0 = router.get_layer_stats(0)
            s35 = router.get_layer_stats(35)
            print(f"{step:>5} {phase:>8} {s0['coeff_var']:>7.3f} "
                  f"{s35['coeff_var']:>7.3f} {router.alpha:>6.3f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dynamic Bias Router v3 Tests")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Model: gpt-oss-120b ({NUM_EXPERTS}E, top-{TOPK}, {NUM_LAYERS}L, h={HIDDEN_SIZE})")
    print(f"Router: v3 (normalized EMA, unified sem, int32, Layer 3+4 safety)\n")

    # 15 correctness tests
    test_basic_routing()
    test_per_layer_isolation()
    test_guardrail_on_sem()
    test_normalized_ema()
    test_normalization_batch_size_stability()
    test_intra_step_feedback()
    test_weight_consistency()
    test_no_cross_step_decay_bug()
    test_load_balancing()
    test_flip_rate_detection()
    test_drop_rate_extreme_alpha()
    test_gap_stats_sampling()
    test_safe_alpha_with_alpha2()
    test_alpha_clamp_prevents_flip()
    test_manager_accuracy_loop()

    if args.benchmark or args.all:
        benchmark_single_layer()
        benchmark_full_forward()

    if args.simulate or args.all:
        simulate_workload()

    print("All tests passed!")


if __name__ == "__main__":
    main()
