# Dynamic Tensor/Layer GPU Placement Proposal (Final)

## Executive Summary

This document proposes an extension to llama.cpp that allows model weight tensors — individually or grouped by layer — to be migrated between CPU and GPU at runtime, controlled via an HTTP API. The goal is to enable ML-based residency prediction (e.g., which MoE experts or dense layers should live in VRAM for a given workload), improving decode throughput or reducing VRAM pressure compared to a static `--n-gpu-layers` value.

Key design decisions (v2, replacing the earlier draft):

1. **No shadow copy.** The existing mmap of the GGUF file is the master copy. GPU→CPU migration is a zero-copy re-point at mapped pages; CPU→GPU is a copy from those pages.
2. **Buffer swap, not tensor replacement.** The `ggml_tensor` object is kept; only its `buffer` and `data` fields are swapped. No pointer chasing through `llama_layer`, `tensors_by_name`, or LoRA references.
3. **Preallocated GPU layer-slot pool.** Fixed-size slots eliminate VRAM fragmentation and allocation cost on the migration path.
4. **Async copy + quiescent-point swap.** Weight upload runs on a side stream while decode continues on the old copy; the swap and scheduler re-reserve happen between decode calls via the server task queue. No model-wide mutex.
5. **Tensor-level granularity.** The internal API migrates arbitrary named tensors (`migrate_tensor`); "layer" is a convenience batch on top. This matches MoE reality where the useful unit is the expert tensors, and mirrors the existing static `--override-tensor` / `--n-cpu-moe` placement machinery.

**Status**: Not implemented. Design proposal.

## Background: How Placement Works Today

`--n-gpu-layers` is a **load-time** parameter (`llama_model_params.n_gpu_layers`), consumed exactly once in `src/llama-model.cpp` during `llama_model_base::load_tensors()`:

```cpp
const int i_gpu_start     = std::max(n_layer_all + 1 - n_gpu_layers, 0);
const int act_gpu_layers  = devices.empty() ? 0 : std::min(n_gpu_layers, n_layer_all + 1);
```

For every layer `il`, `get_layer_buft_list(il)` returns either the CPU buffer-type list or a GPU buffer-type list. Weight tensors are created with that buffer type and uploaded during load. After load:

- `llama_model::impl::dev_layer[il]` stores the assigned backend device.
- `llama_model::impl::ctxs_bufs` holds the allocated backend buffers.
- `llama_model::tensors_by_name` maps names to `ggml_tensor*` pointers.
- Per-architecture `layers[il]` structs contain direct `ggml_tensor*` pointers to layer weights.
- `llama_context` builds a compute graph; `ggml_backend_sched` places ops based on where weight tensors live.

Tensor-level static placement already exists (`--override-tensor` regex, `--n-cpu-moe`), but there is no public API to change any placement after `llama_load_model_from_file()` returns.

**Important baseline**: `ggml_backend_sched` already performs per-op weight upload for CPU-resident tensors when the batch size exceeds a threshold — this is why prompt processing remains fast with partial offload. Dynamic residency therefore primarily benefits **token generation (batch=1)**, where per-op upload does not trigger. Any ML residency predictor should optimize decode throughput specifically, and the per-op offload path is the baseline it must beat.

## Goal

Expose an HTTP endpoint (and matching internal API) that changes tensor/layer placement while the server is running:

```bash
POST /v1/model/tensor-placement
{
  "changes": [
    { "layer": 15, "backend": "gpu", "device": 0 },
    { "tensor": "blk.20.ffn_gate_exps.weight", "backend": "cpu" }
  ]
}
```

After the call completes, subsequent `llama_decode()` calls execute the affected ops on the chosen backends. Changes are **batched**: one request may move many tensors and triggers exactly one scheduler re-reserve.

## Design

### 1. mmap as the master copy (no shadow copy)

llama.cpp loads with `use_mmap=true` by default. CPU-resident tensor `data` pointers already point directly into the mapped GGUF pages; GPU upload at load time is a copy *from* those pages. We exploit this instead of duplicating weights:

- When `use_dynamic_offload` is enabled, keep the `llama_model_loader` mapping (and per-tensor file offsets) alive for the model's lifetime.
- **GPU→CPU migration**: re-point the tensor's `data` at the mmap'd region. Zero-copy; the page cache does the work, and cold pages fault in lazily.
- **CPU→GPU migration**: `cudaMemcpy` (or backend equivalent) from the mmap'd pages into the target GPU slot.
- RAM cost is page-cache pressure only — reclaimable by the kernel — not a doubled RSS.

**Constraints**:

- Requires `use_mmap=true` and the model file remaining available for the process lifetime. If the model is loaded from a temporary/remote source without mmap, dynamic mode is unavailable (fail at load with a clear error) — we deliberately do not fall back to a heap shadow copy in v1.
- **CPU repack/extra-buffer-type backends (repack, AMX) are incompatible**: they transform weights at load into a different in-memory layout, so a repacked CPU tensor cannot be re-pointed at raw GGUF bytes, and raw GGUF bytes are not what repacked kernels expect. `use_dynamic_offload` disables weight repacking (log a notice about the CPU-side performance impact).

### 2. Buffer swap instead of tensor replacement

The earlier draft proposed creating a new tensor on the target backend and updating every reference. That is the riskiest possible strategy and unnecessary. For standard contiguous weight tensors, `ggml_tensor::nb` strides derive from type + shape, not from the backend. Therefore:

- Keep the `ggml_tensor` object.
- Allocate (or check out, see §3) space in the target backend buffer.
- Copy data in.
- Swap `tensor->buffer` and `tensor->data` in place.

Every existing reference — `llama_layer` members, `tensors_by_name`, LoRA adapter base-weight pointers — continues to point at the same `ggml_tensor` and remains valid. No pointer chasing, no crash risk from a missed reference.

**Guard**: before migrating, assert the target buffer type's `get_alloc_size` and alignment match the tensor's current layout. This holds for CUDA and plain CPU buffers; it does not hold for repacked CPU layouts (excluded per §1).

### 3. GPU layer-slot pool

Decoder layers are homogeneous: same tensor set, same sizes. To eliminate fragmentation and allocation latency:

- At startup, preallocate **K GPU layer slots**, each sized `max(per-layer weight footprint)`, where K = the maximum number of GPU-resident layers permitted (configurable; defaults to fitting available VRAM after KV/compute buffers).
- Migration to GPU checks a slot out of the free list; migration off GPU checks it back in.
- For tensor-granular MoE mode, slots are sub-allocated per tensor group (e.g., expert-tensor-sized sub-slots), or a second pool with expert-sized slots is maintained — same principle.

Benefits: zero VRAM fragmentation over arbitrarily many migrations, O(1) allocation on the migration path, and a hard, predictable VRAM budget instead of hoping repeated `alloc`/`free` converges.

### 4. Migration execution: async copy + quiescent-point swap

A model-wide mutex around `llama_decode()` would stall inference for the full copy duration. Instead, migration is split into two phases:

**Phase A — async copy (overlaps decode):**
Weights are read-only during inference, so copying the *source* data into the *target* slot on a side stream (e.g., a dedicated CUDA stream) is safe while decode continues using the old copy. For GPU→CPU, this phase is trivial (the mmap data already exists; optionally `madvise(MADV_WILLNEED)` to prefault pages).

**Phase B — swap at quiescent point (microseconds):**
The server has a single decode loop with a task queue. Migration is injected as a **server task**: between decode calls, on the decode thread itself, perform the `buffer`/`data` swap for all tensors in the batch, update `dev_layer[]`, return freed slots to the pool, and trigger one scheduler re-reserve. Because this runs on the decode thread at a natural quiescent point, **no new locking is required**.

This also resolves KV-cache safety for free: no layer's attention op is ever mid-graph when its weights swap.

**Cost model**: PCIe 4.0 x16 sustains ~25 GB/s in practice, so a ~500 MB layer copies in ~20 ms — and that copy overlaps decode. The visible stall is only the swap + re-reserve. A full 32-layer reshuffle is a few hundred ms of background copy. Per-request/per-session adaptation is comfortably viable; per-token residency changes remain out of scope.

### 5. Runtime placement table

Add to `llama_model::impl`:

```cpp
struct llama_model::impl {
    // ... existing members ...

    // Runtime placement: tensor name -> current backend device.
    // Populated at load from the static n_gpu_layers / override-tensor result;
    // mutated only at quiescent points by the migration task.
    std::unordered_map<std::string, ggml_backend_dev_t> tensor_placement;

    // GPU slot pool (per device)
    std::vector<layer_slot_pool> slot_pools;

    // Registered contexts to notify on placement change
    std::vector<llama_context *> contexts;  // register/unregister in ctx ctor/dtor
};
```

### 6. Migration API (internal)

```cpp
// Tensor-granular primitive. "layer" migration is a batch of these.
// Executed in two phases as described in §4; this is the synchronous
// entry point used by the server task.
bool llama_model::migrate_tensors(
        const std::vector<std::pair<std::string, ggml_backend_dev_t>> & changes,
        std::string & error);
```

Steps per batch:

1. **Validate** every change: tensor exists; target device supports the op/quant type (probe with `ggml_backend_dev_supports_op` on a representative mul_mat for the tensor's type); slot/sub-slot available in the pool. Reject the whole batch with a reason on any failure — placement changes are atomic per request.
2. **Net-out no-ops**: if the batch nets to no placement change for a tensor (CPU→GPU0 then GPU0→CPU), drop it. If the *entire* batch nets to zero, skip the re-reserve.
3. **Phase A**: async-copy all CPU→GPU tensors into their checked-out slots; prefault mmap pages for GPU→CPU tensors.
4. **Phase B** (decode-thread task): swap `buffer`/`data` pointers, update `tensor_placement` and `dev_layer[]`, return freed slots, notify all registered contexts.

### 7. Context re-reserve

After placement changes, the existing `ggml_backend_sched` graph partition is invalid. Add:

```cpp
void llama_context::notify_model_backends_changed();
```

Implementation: set `sched_need_reserve = true`; re-reserve happens on the next `llama_decode()` (mirroring how context-size changes already trigger re-reserve). Because migrations are batched, N tensor moves cost exactly one re-reserve. All contexts in the model's registry are notified (multi-context safety), though the server typically runs one.

### 8. HTTP API

```
GET  /v1/model/tensor-placement    -> current placement map (per layer + per overridden tensor)
POST /v1/model/tensor-placement    -> batched placement changes
```

POST request body:

```json
{
  "changes": [
    { "layer": 15, "backend": "gpu", "device": 0 },
    { "tensor": "blk.20.ffn_gate_exps.weight", "backend": "cpu" }
  ]
}
```

- `layer` expands server-side to all weight tensors of that layer (convenience sugar over the tensor primitive).
- `backend`: `"cpu"` or `"gpu"`; `device` selects the GPU index (default 0).

Response:

```json
{
  "success": true,
  "applied": 2,
  "noops": 0,
  "reserve_triggered": true
}
```

On failure: `{ "success": false, "error": "blk.20.ffn_gate_exps.weight: type q4_K not supported on device CUDA1" }` with HTTP 400/409.

The handler enqueues a migration task on the server task queue and awaits its result — it does not touch model state directly.

Registered in `tools/server/server.cpp`:

```cpp
ctx_http.get ("/v1/model/tensor-placement", ex_wrapper(routes.get_tensor_placement));
ctx_http.post("/v1/model/tensor-placement", ex_wrapper(routes.post_tensor_placement));
```

### 9. MoE focus

For MoE models, the high-value unit is the expert tensors (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`), not whole layers: the standard split keeps attention + shared experts on GPU and routed experts on CPU (what `--n-cpu-moe` does statically today). The tensor-granular API makes the dynamic version of this trivial, and per-expert residency driven by routing statistics is precisely the ML-prediction use case motivating this proposal. Nothing MoE-specific is needed in the core mechanism — only in the predictor that calls the API.

## Files Requiring Changes

| File | Changes |
|------|---------|
| `src/llama-model.h` | `migrate_tensors()` declaration; placement table and slot-pool members; context registry. |
| `src/llama-model.cpp` | Keep mmap alive under `use_dynamic_offload`; disable CPU repack in this mode; slot pool; `migrate_tensors()` (validate / net-out / async copy / swap); populate `tensor_placement` at load. |
| `src/llama-model-loader.cpp` / `.h` | Expose per-tensor mmap offsets and keep mapping alive for model lifetime in dynamic mode. |
| `src/llama-context.h` / `.cpp` | `notify_model_backends_changed()`; context register/unregister with model; re-reserve on next decode. |
| `include/llama.h` (optional) | Public C API: `llama_model_migrate_tensors()` if external embedders need it. |
| `tools/server/server-context.h` / `.cpp` | GET/POST handlers; migration as a server task executed at the decode-loop quiescent point. |
| `tools/server/server.cpp` | Register `/v1/model/tensor-placement` routes. |

## Risks and Open Questions

1. **Layout-transforming CPU backends.** Repack/AMX are disabled in dynamic mode (§1). Acceptable trade-off, but should be measured — for some quants CPU repack matters.
2. **Quantization backend support.** Handled by per-batch `ggml_backend_dev_supports_op` validation with a descriptive HTTP error.
3. **mmap availability.** Dynamic mode requires mmap + persistent model file. No-mmap fallback (heap master copy) is deferred; fail loudly at load.
4. **LoRA adapters.** Buffer-swap keeps base tensor objects stable, so adapter references remain valid. Open question: adapters whose *own* tensors were placed relative to base placement may want to follow migrations — out of scope for v1, documented limitation.
5. **KV cache / Flash Attention.** Weight migration does not touch KV. Quiescent-point swap guarantees no mid-graph changes. If per-layer KV placement is added later, FA kernel availability per backend must be re-validated.
6. **Slot sizing for heterogeneous layers.** Some architectures have non-uniform layers (e.g., first/last layer extras, hybrid attention). Slot size = max over migratable layers; small internal waste, or maintain per-size pools.
7. **Scheduler re-reserve cost.** Re-reserve is not free (graph re-partition + compute-buffer re-alloc). Batching bounds it to one per API call; measure on large graphs.
8. **Concurrent API calls.** Serialize migration tasks in the server queue (natural FIFO); a second request queues behind the first.

## Implementation Phases

### Phase 1: mmap retention + placement table
- `use_dynamic_offload` flag: keep loader mapping alive, record per-tensor offsets, disable CPU repack, populate `tensor_placement` at load.

### Phase 2: Buffer-swap primitive
- Single-tensor swap (`buffer`/`data` in place) with layout/alignment guard.
- Unit test: swap a tensor CPU↔GPU, run mul_mat via `test-backend-ops`, verify bit-identical results vs static placement.

### Phase 3: Slot pool
- Per-device layer-slot pool with check-out/check-in; sizing from load-time layer footprints.

### Phase 4: Batched migration
- `migrate_tensors()`: validation, net-out, Phase-A async copy, Phase-B swap. Layer→tensor expansion.

### Phase 5: Context notification
- Context registry on model; `notify_model_backends_changed()`; one re-reserve per batch; skip when batch nets to zero.

### Phase 6: Server API
- GET/POST `/v1/model/tensor-placement`; migration as decode-loop task; error reporting.

### Phase 7: Validation
- Existing server tests.
- Mid-conversation migration test: identical outputs before/after migration (greedy sampling), no VRAM growth over 1000 migration cycles (fragmentation check), decode-latency histogram during background copy (stall check).

## Alternative Considered: Whole-Model Reload

Reloading with a different `n_gpu_layers` achieves the same user-facing goal at the cost of full reload latency (seconds to minutes) and loss of KV cache. Rejected as the primary mechanism; remains the fallback for configurations where dynamic mode is unavailable (no mmap, repack-critical CPU deployments).

## Conclusion

With the mmap-as-master-copy, in-place buffer swap, slot pool, and quiescent-point execution model, dynamic placement requires **no RAM duplication, no pointer-replacement risk, no VRAM fragmentation, and no decode stalls beyond a microsecond-scale swap plus one scheduler re-reserve per batch**. The mechanism is tensor-granular, making it directly applicable to MoE expert residency — the highest-value use case — while layer-level control falls out as a convenience layer. Implementation should proceed in the listed phases with bit-exactness and fragmentation tests gating each step.
