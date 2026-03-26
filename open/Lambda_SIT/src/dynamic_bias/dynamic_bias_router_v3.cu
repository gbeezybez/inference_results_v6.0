
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cstdint>
#include <cstring>

// ---------------------------------------------------------------------------
// Configuration (gpt-oss-120b)
// ---------------------------------------------------------------------------
constexpr int NUM_EXPERTS = 128;
constexpr int TOPK = 4;
constexpr int NUM_LAYERS = 36;

// Device params indices
constexpr int PARAM_ALPHA = 0;
constexpr int PARAM_GUARDRAIL_THR = 1;
constexpr int PARAM_ALPHA2 = 2;  // intra-step feedback strength
constexpr int NUM_PARAMS = 3;


// ---------------------------------------------------------------------------
// Main Kernel: Dynamic Bias Router v3
// ---------------------------------------------------------------------------
// One block per token, 128 threads (1 thread per expert).
// All intermediate data in shared memory.
//
// Semantic baseline: sem[e] = raw[e] + bias_base[e]
//   - This is what the model "would have computed" without dynamic bias.
//   - Guardrail gap uses sem.
//   - Routing score = sem - α*load_ema (- α2*counts if enabled).
//   - Weights = softmax(sem[selected]) — same for locked and unlocked.
// ---------------------------------------------------------------------------
__global__ void dynamic_bias_router_v3_kernel(
    const float* __restrict__ logits,       // [num_tokens, NUM_EXPERTS]
    const float* __restrict__ bias_base,    // [NUM_EXPERTS]
    const float* __restrict__ load_ema,     // [NUM_EXPERTS] (this layer, normalized)
    int32_t*                   counts,      // [NUM_EXPERTS] read for α2 + atomicAdd target
    const float* __restrict__ params,       // [NUM_PARAMS] device scalars
    int32_t*     __restrict__ out_indices,  // [num_tokens, TOPK]
    float*       __restrict__ out_weights,  // [num_tokens, TOPK]
    int32_t*     __restrict__ out_flags,    // [num_tokens]
    int32_t*                   flip_cnt,    // [1] per-layer flip counter (nullable)
    int32_t*                   lock_cnt,    // [1] per-layer lock counter (nullable)
    int32_t*                   drop_cnt,    // [1] per-layer top1-out-of-top4 counter (nullable)
    const int    num_tokens,
    const int    enable_accuracy_stats      // 0=skip counters, 1=count (sampling flag)
) {
    const int token_idx = blockIdx.x;
    const int eid = threadIdx.x;  // expert id, 0..127

    if (token_idx >= num_tokens || eid >= NUM_EXPERTS) return;

    // Read device-side parameters (CUDA Graph safe)
    __shared__ float s_alpha;
    __shared__ float s_guardrail_thr;
    __shared__ float s_alpha2;
    if (eid == 0) {
        s_alpha = params[PARAM_ALPHA];
        s_guardrail_thr = params[PARAM_GUARDRAIL_THR];
        s_alpha2 = params[PARAM_ALPHA2];
    }

    // ---- Shared memory layout ----
    __shared__ float s_sem[NUM_EXPERTS];        // semantic baseline: raw + bias
    __shared__ float s_scoring[NUM_EXPERTS];    // routing scores (modified during top-k)
    __shared__ int   s_topk_idx[TOPK];
    __shared__ int   s_is_locked;

    __syncthreads();

    // ---- Step 1: All threads compute in parallel ----
    const int offset = token_idx * NUM_EXPERTS + eid;
    const float raw = logits[offset];
    const float bias = bias_base[eid];
    const float load = load_ema[eid];            // normalized: 1.0 = average
    const float cur_count = (float)counts[eid];  // intra-step (stale but useful)

    // Semantic baseline: what the model would compute without dynamic bias
    const float sem = raw + bias;
    s_sem[eid] = sem;

    // Routing score: sem minus load penalties
    float score = sem - s_alpha * load;
    if (s_alpha2 > 0.0f) {
        score -= s_alpha2 * cur_count;  // intra-step burst damping
    }
    s_scoring[eid] = score;

    __syncthreads();

    // ---- Steps 2-4: Thread 0 does guardrail + top-k + weights ----
    if (eid == 0) {
        // -- Step 2: Guardrail on SEMANTIC baseline (not raw alone) --
        float top1_val = -FLT_MAX, top2_val = -FLT_MAX;
        int top1_idx = 0;

        #pragma unroll 4
        for (int e = 0; e < NUM_EXPERTS; e++) {
            float v = s_sem[e];
            if (v > top1_val) {
                top2_val = top1_val;
                top1_val = v;
                top1_idx = e;
            } else if (v > top2_val) {
                top2_val = v;
            }
        }

        const float gap = top1_val - top2_val;
        const int locked = (gap > s_guardrail_thr) ? 1 : 0;
        s_is_locked = locked;

        // -- Step 3: Iterative top-k (no bitmask, just -FLT_MAX marking) --
        for (int k = 0; k < TOPK; k++) {
            int best_idx;

            if (k == 0 && locked) {
                // Guardrail: force top-1 from semantic baseline
                best_idx = top1_idx;
            } else {
                // Argmax on s_scoring
                float best_val = -FLT_MAX;
                best_idx = 0;

                #pragma unroll 4
                for (int e = 0; e < NUM_EXPERTS; e++) {
                    if (s_scoring[e] > best_val) {
                        best_val = s_scoring[e];
                        best_idx = e;
                    }
                }
            }

            s_topk_idx[k] = best_idx;
            s_scoring[best_idx] = -FLT_MAX;  // exclude from future picks
        }

        // -- Step 4: Weights = softmax(sem[selected]) --
        // SAME for locked and unlocked. No load penalty in weights.
        float w_logits[TOPK];
        float max_val = -FLT_MAX;

        #pragma unroll
        for (int k = 0; k < TOPK; k++) {
            w_logits[k] = s_sem[s_topk_idx[k]];
            max_val = fmaxf(max_val, w_logits[k]);
        }

        float sum_exp = 0.0f;
        float exps[TOPK];
        #pragma unroll
        for (int k = 0; k < TOPK; k++) {
            exps[k] = expf(w_logits[k] - max_val);
            sum_exp += exps[k];
        }

        const float inv_sum = 1.0f / sum_exp;
        const int out_base = token_idx * TOPK;

        #pragma unroll
        for (int k = 0; k < TOPK; k++) {
            out_indices[out_base + k] = (int32_t)s_topk_idx[k];
            out_weights[out_base + k] = exps[k] * inv_sum;
        }
        out_flags[token_idx] = locked;

        // ---- Accuracy monitoring (guarded by sampling flag to avoid contention) ----
        if (enable_accuracy_stats) {
            // top1_idx = semantic top-1 (from guardrail check on sem)
            // s_topk_idx[0] = final selected top-1 (from adjusted scores)
            if (lock_cnt != nullptr) {
                atomicAdd(lock_cnt, locked);
            }
            if (flip_cnt != nullptr && !locked && s_topk_idx[0] != top1_idx) {
                atomicAdd(flip_cnt, 1);
            }
            // Drop = semantic top-1 is NOT in the final top-4 at all
            // This is the hard safety metric for Layer 3
            if (drop_cnt != nullptr) {
                int top1_in_topk = 0;
                #pragma unroll
                for (int k = 0; k < TOPK; k++) {
                    if (s_topk_idx[k] == top1_idx) top1_in_topk = 1;
                }
                if (!top1_in_topk) {
                    atomicAdd(drop_cnt, 1);
                }
            }
        }
    }

    __syncthreads();

    // ---- Step 5: Atomic count update (only TOPK threads) ----
    if (eid < TOPK) {
        atomicAdd(&counts[s_topk_idx[eid]], 1);
    }
}


// ---------------------------------------------------------------------------
// EMA Update Kernel v3 (runs ONCE per decode step, all layers)
// ---------------------------------------------------------------------------
// Key change: normalizes counts to relative-to-mean before EMA.
//   norm_count = raw_count * (NUM_EXPERTS / (num_tokens * TOPK))
//   So average expert → norm_count ≈ 1.0
//   α now means "per unit of average load, penalize by α logits"
// ---------------------------------------------------------------------------
__global__ void ema_update_all_layers_v3_kernel(
    float*   __restrict__ load_ema,         // [total_elements]
    int32_t* __restrict__ counts,           // [total_elements] (raw or allreduced)
    const float beta,                       // EMA smoothing
    const float inv_mean,                   // NUM_EXPERTS / (num_tokens * TOPK)
    const int total_elements                // NUM_LAYERS * NUM_EXPERTS
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    const float old_ema = load_ema[idx];
    const float raw_count = (float)counts[idx];

    // Normalize: 1.0 = average load
    const float norm_count = raw_count * inv_mean;

    // EMA update on normalized load
    load_ema[idx] = beta * old_ema + (1.0f - beta) * norm_count;

    // Reset count for next step
    counts[idx] = 0;
}


// ---------------------------------------------------------------------------
// C API
// ---------------------------------------------------------------------------
extern "C" {

struct DynamicBiasRouterV3 {
    // Per-layer state, flattened as [NUM_LAYERS, NUM_EXPERTS]
    int32_t* counts;        // device: per-step atomic counts
    float*   load_ema;      // device: normalized smoothed load (1.0 = avg)

    // Device-side params (CUDA Graph safe — update via async memcpy)
    float*   d_params;      // device: [alpha, guardrail_threshold, alpha2]

    // Host mirror for convenience
    float    h_params[NUM_PARAMS];

    // Accuracy monitoring counters: [NUM_LAYERS] each
    int32_t* flip_cnt;      // device: per-layer top-1 flip count (soft metric)
    int32_t* lock_cnt;      // device: per-layer locked token count
    int32_t* drop_cnt;      // device: per-layer top-1-out-of-top4 count (hard safety)

    // Sampling control: avoid atomicAdd contention on large prefill batches
    int   accuracy_stats_enabled;  // 0=off, 1=on (toggle every N steps)

    // Config (immutable after creation)
    int   num_layers;
    int   num_experts;
    int   topk;
    float ema_beta;

    // Stats
    int   step_count;
    int   last_num_tokens;  // for normalization
};

DynamicBiasRouterV3* create_router_v3(
    int   num_layers,            // 36
    int   num_experts,           // 128
    int   topk,                  // 4
    float alpha,                 // 0.1 - 0.3
    float guardrail_threshold,   // 1.5 - 2.5
    float ema_beta,              // 0.8 - 0.9
    float alpha2                 // 0.0 = disabled, 0.01-0.05 = intra-step
) {
    auto* r = new DynamicBiasRouterV3();
    r->num_layers = num_layers;
    r->num_experts = num_experts;
    r->topk = topk;
    r->ema_beta = ema_beta;
    r->step_count = 0;
    r->last_num_tokens = 0;

    // Host params
    r->h_params[PARAM_ALPHA] = alpha;
    r->h_params[PARAM_GUARDRAIL_THR] = guardrail_threshold;
    r->h_params[PARAM_ALPHA2] = alpha2;

    // Device allocations
    const int total = num_layers * num_experts;  // 4608
    cudaMalloc(&r->counts, total * sizeof(int32_t));
    cudaMalloc(&r->load_ema, total * sizeof(float));
    cudaMalloc(&r->d_params, NUM_PARAMS * sizeof(float));

    // Accuracy monitoring counters
    cudaMalloc(&r->flip_cnt, num_layers * sizeof(int32_t));
    cudaMalloc(&r->lock_cnt, num_layers * sizeof(int32_t));
    cudaMalloc(&r->drop_cnt, num_layers * sizeof(int32_t));

    cudaMemset(r->counts, 0, total * sizeof(int32_t));
    cudaMemset(r->load_ema, 0, total * sizeof(float));
    cudaMemset(r->flip_cnt, 0, num_layers * sizeof(int32_t));
    cudaMemset(r->lock_cnt, 0, num_layers * sizeof(int32_t));
    cudaMemset(r->drop_cnt, 0, num_layers * sizeof(int32_t));
    cudaMemcpy(r->d_params, r->h_params, NUM_PARAMS * sizeof(float),
               cudaMemcpyHostToDevice);

    r->accuracy_stats_enabled = 0;  // off by default, host turns on for sampling windows

    return r;
}

void destroy_router_v3(DynamicBiasRouterV3* r) {
    if (r) {
        cudaFree(r->counts);
        cudaFree(r->load_ema);
        cudaFree(r->d_params);
        cudaFree(r->flip_cnt);
        cudaFree(r->lock_cnt);
        cudaFree(r->drop_cnt);
        delete r;
    }
}

/**
 * Route tokens for a single MoE layer.
 * Call 36 times per decode step (once per layer).
 * Only does routing + atomicAdd to counts. NO EMA here.
 *
 * NOTE: out_indices is int32_t (not int64_t) for TRT-LLM compat.
 */
void router_v3_forward(
    DynamicBiasRouterV3* r,
    int           layer_idx,        // 0..35
    const float*  logits,           // device [num_tokens, 128]
    const float*  bias_base,        // device [128]
    int32_t*      out_indices,      // device [num_tokens, 4]  ← int32
    float*        out_weights,      // device [num_tokens, 4]
    int32_t*      out_flags,        // device [num_tokens]
    int           num_tokens,
    cudaStream_t  stream
) {
    const int layer_offset = layer_idx * r->num_experts;
    float*   layer_ema    = r->load_ema + layer_offset;
    int32_t* layer_counts = r->counts + layer_offset;

    dim3 grid(num_tokens);
    dim3 block(NUM_EXPERTS);  // 128 threads

    dynamic_bias_router_v3_kernel<<<grid, block, 0, stream>>>(
        logits, bias_base, layer_ema, layer_counts,
        r->d_params,    // device-side α, threshold, α2
        out_indices, out_weights, out_flags,
        r->flip_cnt + layer_idx,   // per-layer flip counter
        r->lock_cnt + layer_idx,   // per-layer lock counter
        r->drop_cnt + layer_idx,   // per-layer drop counter (top1 out of top4)
        num_tokens,
        r->accuracy_stats_enabled  // sampling flag
    );

    // Track for normalization
    r->last_num_tokens = num_tokens;
}

/**
 * Update EMA and reset counts for ALL layers.
 * Call ONCE per decode step after all 36 layers.
 *
 * @param num_tokens_this_step  Total tokens routed this step (for normalization).
 *                              If 0, uses last_num_tokens from forward().
 * @param global_counts         Optional: allreduced counts from all ranks (EP).
 *                              If non-null, replaces local counts before EMA.
 *                              Caller is responsible for NCCL AllReduce.
 */
void router_v3_step(
    DynamicBiasRouterV3* r,
    int           num_tokens_this_step,
    int32_t*      global_counts,    // device, nullable
    cudaStream_t  stream
) {
    const int total = r->num_layers * r->num_experts;  // 4608

    // If caller provides allreduced counts, swap them in
    int32_t* counts_to_use = r->counts;
    if (global_counts != nullptr) {
        counts_to_use = global_counts;
    }

    // Normalization: inv_mean = NUM_EXPERTS / (num_tokens * TOPK)
    // So average expert count maps to 1.0
    int nt = (num_tokens_this_step > 0) ? num_tokens_this_step : r->last_num_tokens;
    if (nt <= 0) nt = 1;  // safety
    const float inv_mean = (float)r->num_experts / ((float)nt * (float)r->topk);

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;  // 18

    ema_update_all_layers_v3_kernel<<<blocks, threads, 0, stream>>>(
        r->load_ema,
        counts_to_use,
        r->ema_beta,
        inv_mean,
        total
    );

    // If we used global_counts, still need to reset local counts
    if (global_counts != nullptr) {
        cudaMemsetAsync(r->counts, 0, total * sizeof(int32_t), stream);
    }

    r->step_count++;
}

/**
 * Hard reset all state.
 */
void router_v3_reset(DynamicBiasRouterV3* r) {
    const int total = r->num_layers * r->num_experts;
    cudaMemset(r->counts, 0, total * sizeof(int32_t));
    cudaMemset(r->load_ema, 0, total * sizeof(float));
    cudaMemset(r->flip_cnt, 0, r->num_layers * sizeof(int32_t));
    cudaMemset(r->lock_cnt, 0, r->num_layers * sizeof(int32_t));
    cudaMemset(r->drop_cnt, 0, r->num_layers * sizeof(int32_t));
    r->step_count = 0;
}

/**
 * Update device-side parameters (CUDA Graph safe: async memcpy, no graph rebuild).
 */
void router_v3_set_alpha(DynamicBiasRouterV3* r, float alpha, cudaStream_t stream) {
    r->h_params[PARAM_ALPHA] = alpha;
    cudaMemcpyAsync(r->d_params + PARAM_ALPHA, &r->h_params[PARAM_ALPHA],
                    sizeof(float), cudaMemcpyHostToDevice, stream);
}

void router_v3_set_guardrail(DynamicBiasRouterV3* r, float threshold, cudaStream_t stream) {
    r->h_params[PARAM_GUARDRAIL_THR] = threshold;
    cudaMemcpyAsync(r->d_params + PARAM_GUARDRAIL_THR, &r->h_params[PARAM_GUARDRAIL_THR],
                    sizeof(float), cudaMemcpyHostToDevice, stream);
}

void router_v3_set_alpha2(DynamicBiasRouterV3* r, float alpha2, cudaStream_t stream) {
    r->h_params[PARAM_ALPHA2] = alpha2;
    cudaMemcpyAsync(r->d_params + PARAM_ALPHA2, &r->h_params[PARAM_ALPHA2],
                    sizeof(float), cudaMemcpyHostToDevice, stream);
}

/**
 * Bulk update all params at once.
 */
void router_v3_set_params(DynamicBiasRouterV3* r, float alpha, float thr,
                          float alpha2, cudaStream_t stream) {
    r->h_params[PARAM_ALPHA] = alpha;
    r->h_params[PARAM_GUARDRAIL_THR] = thr;
    r->h_params[PARAM_ALPHA2] = alpha2;
    cudaMemcpyAsync(r->d_params, r->h_params, NUM_PARAMS * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
}

/**
 * Copy load_ema to host for monitoring.
 * Caller must free() returned buffer.
 */
float* router_v3_get_load_ema(DynamicBiasRouterV3* r, int layer_idx) {
    const int n = r->num_experts;
    float* host_buf = (float*)malloc(n * sizeof(float));
    if (!host_buf) return nullptr;
    cudaMemcpy(host_buf, r->load_ema + layer_idx * n,
               n * sizeof(float), cudaMemcpyDeviceToHost);
    return host_buf;
}

/**
 * Get raw counts (before normalization) for a layer.
 * Useful for intra-step monitoring.
 */
int32_t* router_v3_get_counts(DynamicBiasRouterV3* r, int layer_idx) {
    const int n = r->num_experts;
    int32_t* host_buf = (int32_t*)malloc(n * sizeof(int32_t));
    if (!host_buf) return nullptr;
    cudaMemcpy(host_buf, r->counts + layer_idx * n,
               n * sizeof(int32_t), cudaMemcpyDeviceToHost);
    return host_buf;
}

/**
 * Get accuracy monitoring stats: flip_count, lock_count, drop_count per layer.
 * Returns [flip_cnt, lock_cnt, drop_cnt] for the given layer.
 * Caller must free() returned buffer (size = 3 * int32_t).
 */
int32_t* router_v3_get_accuracy_stats(DynamicBiasRouterV3* r, int layer_idx) {
    int32_t* host_buf = (int32_t*)malloc(3 * sizeof(int32_t));
    if (!host_buf) return nullptr;
    cudaMemcpy(&host_buf[0], r->flip_cnt + layer_idx,
               sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_buf[1], r->lock_cnt + layer_idx,
               sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_buf[2], r->drop_cnt + layer_idx,
               sizeof(int32_t), cudaMemcpyDeviceToHost);
    return host_buf;
}

/**
 * Reset accuracy counters for all layers.
 * Call after reading stats to start a new monitoring interval.
 */
void router_v3_reset_accuracy_stats(DynamicBiasRouterV3* r) {
    cudaMemset(r->flip_cnt, 0, r->num_layers * sizeof(int32_t));
    cudaMemset(r->lock_cnt, 0, r->num_layers * sizeof(int32_t));
    cudaMemset(r->drop_cnt, 0, r->num_layers * sizeof(int32_t));
}
 
/**
 * Enable/disable accuracy stats collection.
 * Use for sampling windows: enable for 2 steps, disable for 48 steps.
 * Avoids atomicAdd contention on large prefill batches.
 */
void router_v3_set_accuracy_stats(DynamicBiasRouterV3* r, int enabled) {
    r->accuracy_stats_enabled = enabled;
}
 
}  // extern "C"
 