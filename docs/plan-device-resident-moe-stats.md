# Plan: Device-Resident MoE Expert Stats

## Problem

`server_moe_router_cb` (`tools/server/server-context.cpp:205`) fires per MoE layer per decode and calls
`ggml_backend_tensor_get` synchronously, causing:

1. **64 device syncs per token** (64-layer model) - each stalls the CPU until the GPU drains.
2. **CUDA graph capture disabled** - any `cb_eval` that reads tensor data forces eager execution (2-5x decode slowdown).
3. **Heap allocation per layer per decode** (`std::vector<uint8_t> data(n_bytes)`).
4. **Unbuffered `printf` per token per layer** on the hot path.
5. **Sample queue bloat** (10k samples, each with string copies).

The rebalancer (`ui/internal/rebalancer/`) consumes aggregated per-expert counts via
`/v1/moe/routed-experts/completion`, not per-token data. It polls at its own interval (seconds,
not per-token).

## Solution

Replace the callback with a **device-resident persistent counter buffer** accumulated in-graph.
The expert selection tensor (`ffn_moe_topk`) already exists as a graph node. After it is computed,
inject a `ggml_get_rows_back` op that scatter-counts selected experts into a persistent device
buffer. The HTTP endpoint reads the buffer on demand (one sync per fetch), then zeros it.

### Why `ggml_get_rows_back` instead of identity + `ggml_get_rows`

The original plan used an identity matrix with `ggml_get_rows` to one-hot encode expert selections.
Review found two fatal issues: (1) `ggml_get_rows` requires `a->ne[2] == b->ne[1]`, so a 2D
identity fails for multi-token batches, and (2) the reshape+sum_rows reduction was
mathematically wrong.

`ggml_get_rows_back` is the scatter-add (histogram) operation we actually need. The CUDA kernel
`k_get_rows_back_float` (`ggml/src/ggml-cuda/getrows.cu:80`) does exactly:
```cpp
for each dst_row in [0, n_expert):
    sum = 0
    for i in [0, n_sel):
        if rows[i] == dst_row:
            sum += grad[i*ncols + col]
    dst[dst_row*ncols + col] = sum
```
With `grad = ones` and `rows = selected_experts`, this counts how many times each expert id
appears. No identity matrix needed.

### Key constraints discovered

- `ggml_get_rows_back` requires `a` (grad) to be a matrix, `b` (indices) to be a **1D vector**
  I32, and `c` (shape ref) to be a matrix. `selected_experts` is 2D `[n_expert_used, n_tokens]`,
  so it must be flattened to 1D via `ggml_view_1d`.
- `ggml_get_rows_back` returns F32 regardless of input types (hardcoded in ggml.c:3893).
- `ggml_acc` / `ggml_acc_inplace` require F32 (asserts `a->type == GGML_TYPE_F32`), requires
  `ggml_is_contiguous(a)`, and asserts `ggml_nelements(b) <= ggml_nelements(a)`.
- `ggml_view_1d` always produces a contiguous tensor (1-D, stride = element size).
- The stats buffer must persist across graph replays at a fixed address (like the KV cache) for
  CUDA graph capture compatibility.
- Zeroing the buffer between graph replays is a host-initiated `ggml_backend_tensor_set` call
  outside the captured graph. This is safe: it's a data mutation between replays, not a
  control-flow change. The graph always does `stats += counts`; the buffer content is just reset.

### Data flow

```
build_moe_ffn (per layer, per decode):
  selected_experts [n_expert_used, n_tokens] I32  (already in graph)
    |
    v  ggml_view_1d (flatten to 1D)
  flat_ids [n_expert_used * n_tokens] I32
    |
    v  ggml_get_rows_back(ones[1, n_sel], flat_ids, shape_ref[1, n_expert])
  counts [1, n_expert] F32  (per-expert counts for this decode)
    |
    v  ggml_view_1d (reshape to 1D)
  counts_1d [n_expert] F32
    |
    v  ggml_acc_inplace(stats_view_il, counts_1d, ...)
  stats_buf[:, il] += counts_1d   (accumulate into persistent device buffer)

HTTP /v1/moe/routed-experts/completion (on demand, every few seconds):
  ggml_backend_tensor_get(stats_buf, host_data, 0, nbytes)  -> one D2H sync
  zero stats_buf via ggml_backend_tensor_set(stats_buf, zeros, 0, nbytes)
  return aggregated JSON
```

## Chunks

### Chunk 1: Device-resident stats buffer (simple)

Allocate a persistent device buffer for accumulated expert counts.

**Files:**
- `src/llama-model.h` - add tensor field
- `src/llama-model.cpp` - allocate buffer

**Interface:**

```cpp
// llama-model.h, inside struct llama_model
struct ggml_tensor * moe_stats_buf = nullptr; // [n_expert, n_layer] F32, device-resident
```

No `moe_identity` tensor is needed (the `ggml_get_rows_back` approach does not use one).

**Allocation (in `llama_model::load_tensors` or a dedicated init method):**

```cpp
// Only allocate for MoE models (n_expert > 0)
if (hparams.n_expert > 0) {
    const int64_t ne_stats[2] = { (int64_t)hparams.n_expert, (int64_t)hparams.n_layer() };
    moe_stats_buf = ggml_new_tensor(ctx_model, GGML_TYPE_F32, 2, ne_stats);
}
```

The tensor gets allocated on the same backend buffer type as the model's main tensors. After
buffer allocation, zero `moe_stats_buf` via `ggml_backend_tensor_set`.

**Zeroing:**

```cpp
std::vector<float> zeros(n_expert * n_layer, 0.0f);
ggml_backend_tensor_set(moe_stats_buf, zeros.data(), 0, zeros.size() * sizeof(float));
```

**Acceptance criteria:**
- `moe_stats_buf` is non-null for MoE models, null for non-MoE models.
- `moe_stats_buf` is zeroed at init (verifiable via `ggml_backend_tensor_get` in a test).
- `moe_stats_buf` resides on the same device as the MoE expert tensors (verifiable via
  `ggml_backend_buft_get_device`).
- Non-MoE models load without changes (tensor is null, accumulation is skipped).

**Edge cases:**
- Non-MoE model: `n_expert == 0` -> skip allocation entirely.
- Multiple devices (tensor split): stats buffer goes on the device that owns the MoE layers
  (use `dev_layer(0)` to pick the buffer type, matching how expert tensors are placed).

---

### Chunk 2: Graph injection of accumulation ops (complex)

Inject accumulation ops into `build_moe_ffn` after `selected_experts` is computed. These ops run
entirely on-device as part of the graph - no callback, no host sync.

**Files:**
- `src/llama-graph.cpp` - inject ops in `build_moe_ffn`
- `src/llama-graph.h` - pass stats buffer via `llm_graph_params`

**Interface changes:**

```cpp
// llama-graph.h - add to llm_graph_params
struct llm_graph_params {
    // ... existing fields ...
    struct ggml_tensor * moe_stats_buf = nullptr; // [n_expert, n_layer] F32, or null
};
```

**Injection point:** `llama-graph.cpp`, in `build_moe_ffn`, after line 1607
(`cb(selected_experts, "ffn_moe_topk", il);`):

```cpp
cb(selected_experts, "ffn_moe_topk", il);

// --- device-resident expert count accumulation (no callback, no D2H sync) ---
// Uses ggml_get_rows_back as a scatter-add (histogram) to count expert selections.
// The CUDA kernel k_get_rows_back_float sums grad[i] for all i where rows[i] == dst_row.
// With grad = ones, this counts occurrences of each expert id.
if (moe_stats_buf != nullptr) {
    const int64_t n_sel = n_expert_used * n_tokens;

    // Flatten selected_experts: [n_expert_used, n_tokens] -> [n_expert_used * n_tokens] I32
    // ggml_get_rows_back requires b to be a 1D vector (asserts ggml_is_vector(b))
    ggml_tensor * flat_ids = ggml_view_1d(ctx0, selected_experts, n_sel, 0);

    // Ones: [1, n_sel] F32 - gradient values to scatter (all 1.0 for counting)
    // ggml_get_rows_back requires a to be a matrix (asserts ggml_is_matrix(a))
    ggml_tensor * ones_template = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_sel);
    ggml_tensor * ones = ggml_fill(ctx0, ones_template, 1.0f);

    // Shape reference: [1, n_expert] - view into stats_buf, only shape matters
    // ggml_get_rows_back requires c to be a matrix (asserts ggml_is_matrix(c))
    // and a->ne[0] == c->ne[0] (both 1 here)
    // Output will be [c->ne[0], c->ne[1]] = [1, n_expert] F32
    ggml_tensor * shape_ref = ggml_view_2d(ctx0, moe_stats_buf, 1, (int64_t)n_expert,
                                            sizeof(float), 0);

    // Scatter-add: counts[e] = sum(1.0 for each i where flat_ids[i] == e)
    ggml_tensor * counts = ggml_get_rows_back(ctx0, ones, flat_ids, shape_ref);
    // counts: [1, n_expert] F32

    // Reshape to 1D [n_expert] for acc
    ggml_tensor * counts_1d = ggml_view_1d(ctx0, counts, n_expert, 0);

    // View into stats_buf[:, il] at byte offset il * n_expert * sizeof(float)
    ggml_tensor * stats_view = ggml_view_1d(ctx0, moe_stats_buf, n_expert,
                                             il * n_expert * sizeof(float));

    // In-place add: stats_view += counts_1d
    // ggml_acc_inplace(ctx, a, b, nb1, nb2, nb3, offset)
    // a = stats_view [n_expert] 1D contiguous F32
    // b = counts_1d  [n_expert] 1D contiguous F32
    // nb1/nb2/nb3 are b's strides in bytes; for 1D, use total size
    ggml_acc_inplace(ctx0, stats_view, counts_1d,
                     n_expert * sizeof(float),
                     n_expert * sizeof(float),
                     n_expert * sizeof(float),
                     0);
    cb(moe_stats_buf, "ffn_moe_stats_acc", il);
}
// --- end accumulation ---
```

**Graph capture compatibility:**
- `moe_stats_buf` is a persistent device tensor at a fixed address (allocated once in chunk 1).
- The accumulation ops are deterministic graph nodes built during `build_graph`, not injected
  at eval time. CUDA graph capture sees the same op sequence each replay.
- This mirrors how the KV cache works: persistent device buffer, modified in-place each decode,
  fully graph-capturable.
- `ggml_acc_inplace` with `inplace=true` returns `ggml_view_tensor(ctx, a)`, aliasing the view.
  No special assertion blocks graph capture. The CUDA backend implements `GGML_OP_ACC`
  (`ggml/src/ggml-cuda/acc.cu:37`) and `GGML_OP_GET_ROWS_BACK`
  (`ggml/src/ggml-cuda/getrows.cu:279`).

**Gate condition:** The injection is gated on `moe_stats_buf != nullptr`, which is set at model
load time. When `LLAMA_MOE_ROUTER_STATS=0` (or `LLAMA_MOE_EXPERT_MODE=none`), the stats buffer
is not allocated and the ops are skipped - zero overhead, graph capture stays on.

**Acceptance criteria:**
- After one decode with `n_tokens=1`, `moe_stats_buf[:, il]` contains the per-expert counts for
  layer `il` for that decode (verifiable via `ggml_backend_tensor_get` in a test).
- After one decode with `n_tokens > 1` (multi-token batch/prefill), counts correctly aggregate
  across all tokens in the batch.
- After N decodes, counts accumulate (each decode adds to the buffer).
- CUDA graph capture is not disabled when stats collection is active (no `cb_eval` registered).
- When `moe_stats_buf` is null (non-MoE or stats disabled), the accumulation ops are not built
  into the graph - zero overhead.
- Existing MoE tests pass unchanged.

**Edge cases:**
- `n_tokens > 1` (prefill batch): `flat_ids` has `n_expert_used * n_tokens` elements. The
  scatter-add kernel iterates over all of them, correctly counting across all tokens.
- Layer with no MoE (shared expert only): `build_moe_ffn` is not called, so no accumulation for
  that layer. The stats column stays zero, which is correct.
- `n_expert_used == 0`: should not happen (asserted upstream), but the ops would produce zero
  counts, which is harmless.

**Complexity: complex.** This chunk involves graph construction with precise tensor shapes and
strides. The `ggml_acc_inplace` stride parameters must match the view layout exactly. The
`ggml_get_rows_back` requires specific tensor dimensionality (vector b, matrix a and c).
Incorrect shapes will hit assertions at graph build time (fail-fast, not silent corruption).
The build agent must verify shapes using `GGML_ASSERT` or by printing `ne[]` during development.

---

### Chunk 3: HTTP endpoint + callback removal (simple)

Change the HTTP endpoint to read from the device buffer instead of the sample deque. Remove the
`cb_eval` registration so graph capture is restored.

**Files:**
- `tools/server/server-context.cpp` - endpoint handler + callback registration
- `tools/server/server-context.h` - remove callback state if no longer needed

**Endpoint change (`get_moe_routed_experts_for_completion`, ~line 5417):**

```cpp
// Old: read from moe_router_state.cmpl_aggs (populated by callback)
// New: read from model->moe_stats_buf (one D2H sync)

if (model->moe_stats_buf == nullptr) {
    // stats disabled - return empty
    res->data = safe_json_to_str({{"error", "MoE stats disabled"}});
    return;
}

const int64_t n_expert = model->hparams.n_expert;
const int64_t n_layer  = model->hparams.n_layer();
const size_t nbytes = n_expert * n_layer * sizeof(float);

std::vector<float> counts(n_expert * n_layer);
ggml_backend_tensor_get(model->moe_stats_buf, counts.data(), 0, nbytes);

// zero the device buffer for the next accumulation window
// This is a host-initiated write OUTSIDE the captured graph, between graph replays.
// Safe: the graph always does stats += counts; resetting the buffer just starts fresh.
std::vector<float> zeros(n_expert * n_layer, 0.0f);
ggml_backend_tensor_set(model->moe_stats_buf, zeros.data(), 0, nbytes);

// build JSON: per-layer expert counts + backend info (from model tensors, no device read)
json layers = json::array();
for (int il = 0; il < n_layer; il++) {
    // backend info from model->layers[il] tensor residency (same logic as expert_backend_str)
    json backend;
    backend["ffn_down_exps"]    = expert_backend_str(model->layers[il].ffn_down_exps,    model->dev_layer(il));
    backend["ffn_gate_up_exps"] = expert_backend_str(model->layers[il].ffn_gate_up_exps, model->dev_layer(il));
    backend["ffn_up_exps"]      = expert_backend_str(model->layers[il].ffn_up_exps,      model->dev_layer(il));
    backend["ffn_gate_exps"]    = expert_backend_str(model->layers[il].ffn_gate_exps,    model->dev_layer(il));

    std::vector<int> layer_counts(n_expert, 0);
    int total = 0;
    for (int e = 0; e < n_expert; e++) {
        layer_counts[e] = (int) counts[il * n_expert + e];
        total += layer_counts[e];
    }

    json layer_json;
    layer_json["layer"]  = il;
    layer_json["total"]  = total;
    layer_json["counts"] = std::move(layer_counts);
    layer_json["backend"] = std::move(backend);
    layers.push_back(std::move(layer_json));
}

json res_json;
res_json["completion_id"] = cmpl_id;
res_json["n_experts"]     = n_expert;
res_json["layers"]        = std::move(layers);
res->data = res_json.dump();
```

**Callback removal (~line 1421):**

```cpp
// Remove this block entirely:
// {
//     const char * mode  = getenv("LLAMA_MOE_EXPERT_MODE");
//     const char * stats = getenv("LLAMA_MOE_ROUTER_STATS");
//     const bool disabled = ...;
//     if (disabled) { ... } else {
//         params_base.cb_eval = server_moe_router_cb;
//         params_base.cb_eval_user_data = &moe_router_state;
//     }
// }

// Replace with: stats buffer is allocated at model load time when LLAMA_MOE_ROUTER_STATS != "0"
// No cb_eval registration - graph capture stays on.
```

**Keep the per-token streaming endpoint** (`/v1/moe/routed-experts` without `/completion`) as
opt-in via `LLAMA_MOE_ROUTER_STATS=1` (explicitly enables the callback for debug/visualization).
Default: off. This preserves the debugging path without affecting the hot path.

**Acceptance criteria:**
- `/v1/moe/routed-experts/completion` returns aggregated expert counts read from the device buffer.
- After reading, the device buffer is zeroed (next read starts fresh).
- No `cb_eval` callback is registered by default.
- CUDA graph capture is enabled (verifiable via server logs showing graph capture, not eager mode).
- The rebalancer (`ui/internal/rebalancer/`) receives the same JSON shape as before.
- With `LLAMA_MOE_ROUTER_STATS=1`, the per-token callback is registered for debugging.
- Existing server tests pass.

**Edge cases:**
- Concurrent reads: the zero-after-read is not atomic. If two HTTP requests arrive simultaneously,
  one may read partial counts. Acceptable for the rebalancer's use case (it polls at intervals).
  If needed, a mutex around the read+zero can be added.
- Stats disabled (`moe_stats_buf` is null): endpoint returns an error or empty response.
- Non-MoE model: endpoint returns an error or empty response.

---

## Chunk ordering

1. **Chunk 1** (stats buffer) - no dependencies
2. **Chunk 2** (graph injection) - depends on chunk 1 (needs `moe_stats_buf`)
3. **Chunk 3** (endpoint + callback removal) - depends on chunk 1 (needs `moe_stats_buf` accessor)

Chunks 2 and 3 both depend on chunk 1 but are independent of each other. They can run in parallel
after chunk 1 completes.

## What this eliminates

| Current overhead | Eliminated by |
|---|---|
| 64 device syncs per token | chunk 3 (no callback) |
| CUDA graph capture disabled | chunk 3 (no `cb_eval`) |
| Heap alloc per layer per decode | chunk 3 (no callback) |
| `printf` per token per layer | chunk 3 (no callback) |
| Sample queue bloat (10k samples) | chunk 3 (direct buffer read) |

## What this adds

| New overhead | Cost |
|---|---|
| One `ggml_view_1d` + `ggml_fill` + `ggml_get_rows_back` + `ggml_view_1d` + `ggml_acc_inplace` per MoE layer per decode | Negligible (tiny tensors: n_expert=8-256, n_expert_used=2-8, n_tokens=1-4). The expert matmuls are orders of magnitude larger. |
| One D2H sync per HTTP fetch (every few seconds) | Negligible (one sync every few seconds vs 64 per token) |
| Stats buffer in VRAM | n_expert * n_layer * 4 bytes = 64KB for 256 experts, 64 layers. Negligible. |

## Test strategy

- **Chunk 1:** unit test that allocates a model with known n_expert, verifies `moe_stats_buf`
  is zeroed after init.
- **Chunk 2:** integration test that runs one decode on a small MoE model, then reads
  `moe_stats_buf` and verifies the counts match the expected expert selections (compare against
  the old callback output on the same input). Must test both `n_tokens=1` (decode) and
  `n_tokens > 1` (prefill batch) to verify the flatten + scatter-add handles multi-token
  correctly.
- **Chunk 3:** HTTP endpoint test that verifies the JSON response shape matches the old format.
  Verify the buffer is zeroed after read (second read returns zeros until next decode).
- **Graph capture:** verify via server logs that CUDA graph capture is active (not eager mode)
  when stats collection is enabled.
