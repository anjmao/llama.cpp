# Dynamic Per-Expert GPU Placement Proposal

## Executive Summary

This document proposes an extension to llama.cpp that allows **individual MoE experts, in individual layers**, to be placed on GPU or CPU at runtime, driven by observed routing statistics. It builds on the earlier "Dynamic Tensor/Layer GPU Placement" proposal (mmap-as-master-copy, slot pools, async copy + quiescent-point swap) but targets a granularity that proposal cannot reach: in GGUF, all experts of a layer are packed into a single 3D tensor (`blk.N.ffn_{gate,up,down}_exps.weight`, shape `[n_embd, n_ff, n_expert]`), and ggml binds one tensor to one backend buffer. Per-expert placement therefore cannot be expressed as tensor migration — it requires changes to how the MoE FFN is built and executed.

Key design decisions:

1. **Partition pattern, not LRU cache.** Each expert tensor is represented at runtime as two packed partitions: a GPU-resident partition of hot experts and a CPU-resident partition of the rest. `MUL_MAT_ID` runs once per partition with locally remapped expert IDs; partial outputs are summed. There is no per-token miss path, no eviction logic in the hot loop.
2. **Static remap between rebalances.** Placement changes only at controlled intervals. The `global expert id → (partition, local index)` map is a small index tensor updated once per rebalance, so decode-path overhead is one gather.
3. **Expert-sized slot pool.** GPU partitions are backed by preallocated expert-sized slots (extending the layer-slot pool of the prior proposal), giving zero VRAM fragmentation and a hard budget.
4. **Interval rebalancer driven by routing counters.** Router top-k IDs are already available host-side each decode; an EWMA over a sliding window ranks experts per layer, and a rebalance task promotes/demotes experts with hysteresis. Copies run on a side stream (Phase A); pointer/remap updates land at the decode-loop quiescent point (Phase B).
5. **Per-token prefetch is explicitly out of scope.** The router selects experts in the same graph that consumes them; on-demand PCIe transfer (~ms per expert group) dwarfs a batch=1 matvec. All movement is interval-based and overlapped with decode.

**Status**: Not implemented. Design proposal. Supersedes the layer/tensor-granular proposal for MoE models; the tensor-granular mechanism remains valid for dense weights (attention, shared experts, norms).

## Background

### How MoE weights are stored and executed today

- Experts are packed: `blk.N.ffn_gate_exps.weight` is one `ggml_tensor` with `ne = [n_embd, n_ff, n_expert]`. Expert `i` occupies the contiguous byte range `data + i*nb[2] .. data + (i+1)*nb[2]`. Slicing an individual expert out of the file/mmap is trivial arithmetic.
- `GGML_OP_MUL_MAT_ID` consumes the packed tensor plus a tensor of selected expert IDs produced by the router (`ggml_top_k` over router logits) **within the same compute graph**. There is no lookahead: expert identity for token *t* is known only mid-graph for token *t*.
- Placement is per-tensor: `--override-tensor 'exps=CPU'` / `--n-cpu-moe` put the *entire* packed expert tensor on CPU. `ggml_backend_sched` then executes that layer's expert matmul on CPU — activations round-trip over PCIe, weights do not move. This is the baseline to beat at decode.
- At batch=1 both CPU and GPU expert matvecs are memory-bandwidth-bound. The per-activation win of GPU residency is approximately `BW_gpu / BW_cpu_effective`, minus the saved activation round-trip. Expected end-to-end speedup is `hit_rate × Δt` — see Validation.

### Why the prior proposal's primitive is insufficient

The buffer-swap primitive migrates whole tensors. For MoE, the smallest migratable unit would be all `n_expert` experts of one projection in one layer — for a 128-expert model that is 128× coarser than the useful unit. Everything below the graph (data copies, slot pools, quiescent-point execution, HTTP surface) carries over; the new work is in graph construction and the `MUL_MAT_ID` execution path.

### Prior art

- llama.cpp issue #20757 proposes a two-tier GPU expert slot cache with persistent `expert_id → slot` mapping and remapped IDs fed to `MUL_MAT_ID` (LRU/SLRU eviction, admission filters).
- llama.cpp discussion #23324 (Metal PoC) demonstrates on-demand expert paging into a compact slot pool with a CPU sidecar resolving `expert → slot` and `MUL_MAT_ID` running unchanged against the pool.
- Research systems (HOBBIT, MoE Lightning, SMoE, Fiddler-style hybrid execution) converge on the same two viable patterns: reactive expert caching, or static/periodic partitioning with hybrid CPU+GPU execution.

This proposal picks the **partitioning** pattern because the driving use case is a predictive rebalancer operating on session/interval horizons, not per-token reactivity. It composes with, and does not preclude, a later cache-pattern implementation.

## Goal

1. Runtime data structure: per (layer, projection) dual-partition expert storage with a remap table, executable by the existing MoE FFN graph with minimal changes.
2. Rebalancer: usage-counter collection, ranking with hysteresis, batched promote/demote executed with the Phase A/B model.
3. Control surface: HTTP endpoints to read placement/statistics and to set placement explicitly or configure automatic rebalancing.

```bash
POST /v1/model/expert-placement
{
  "changes": [
    { "layer": 20, "experts_gpu": [3, 17, 41, 88], "device": 0 }
  ]
}
```

or automatic:

```bash
POST /v1/model/expert-placement/policy
{ "mode": "auto", "interval_tokens": 512, "window_tokens": 4096,
  "hysteresis": 2, "vram_budget_mb": 6144 }
```

## Design

### 1. Dual-partition expert storage

For each MoE layer `N` and each projection `p ∈ {gate, up, down}` under `use_dynamic_experts`:

- **CPU partition** = the original mmap'd packed tensor, unchanged. It always contains *all* experts (master copy, per the mmap-as-master-copy principle). CPU-side execution indexes it with global IDs directly — no CPU-side compaction, no CPU copies on rebalance.
- **GPU partition** = a packed pseudo-tensor of `K_N` expert-sized slots on the target device, backed by the slot pool (§3). Contains copies of the currently-hot experts for that layer, in slot order.

Runtime metadata per (layer, projection is shared — placement is decided per *expert*, and gate/up/down for that expert move together):

```cpp
struct expert_partition_state {
    int32_t              n_gpu;                  // experts currently on GPU
    std::vector<int32_t> gpu_slot_of_expert;     // [n_expert] -> slot idx or -1
    std::vector<int32_t> expert_of_gpu_slot;     // [K] -> global expert id
    ggml_tensor        * remap;                  // i32 [n_expert], device-resident:
                                                 //   >=0  -> GPU slot index
                                                 //   -1   -> CPU-resident
};
```

Moving gate/up/down together is required for the split-execution scheme (§2): a routed token's FFN for one expert must run entirely on one side, otherwise intermediate activations ping-pong per expert.

### 2. Graph changes: split MUL_MAT_ID

`build_moe_ffn` (src/llama-graph.cpp) currently emits, per layer:

```
ids   = top_k(router_logits)
out   = mul_mat_id(ffn_down_exps, act(mul_mat_id(ffn_gate_exps, x, ids)) * mul_mat_id(ffn_up_exps, x, ids), ids)
```

Under `use_dynamic_experts` this becomes a two-branch computation:

```
ids        = top_k(router_logits)                    # global expert ids
slot_ids   = gather(remap, ids)                      # >=0 gpu slot, -1 cpu
ids_gpu    = where(slot_ids >= 0, slot_ids, SENTINEL)
ids_cpu    = where(slot_ids <  0, ids,      SENTINEL)

out_gpu    = moe_ffn(gpu_partition_tensors, x, ids_gpu)   # runs on GPU
out_cpu    = moe_ffn(cpu_packed_tensors,    x, ids_cpu)   # runs on CPU
out        = weighted_sum(out_gpu, out_cpu, router_probs)
```

Implementation notes:

- **SENTINEL handling.** `MUL_MAT_ID` must skip sentinel rows (produce zeros). Two options: (a) extend the op with a "skip id" convention — small kernel change per backend; (b) keep kernels unchanged and mask the per-expert weights to zero for non-resident experts on each branch, letting the weighted sum drop their contribution. Option (b) computes wasted matvecs for masked experts; option (a) is the target, (b) is an acceptable bring-up path. **Decision: (a)** for CUDA + CPU in v1; other backends fall back to whole-tensor placement.
- **Branch placement.** The GPU branch's weight pseudo-tensors live in GPU buffers, the CPU branch's in the mmap CPU buffer; `ggml_backend_sched` places each branch correctly with no scheduler changes, exactly as it does today for `-ot exps=CPU`.
- **Graph topology is fixed.** Both branches exist in the graph permanently. Rebalances change only the *contents* of `remap` and the GPU partition slots — no re-reserve is needed for expert-only rebalances (contrast: one re-reserve per batch in the tensor-granular proposal). If a layer's `K_N` changes (budget reallocation), that is a partition resize and does trigger a re-reserve.
- **Empty-branch fast path.** If all selected experts for a batch fall in one partition, skip the other branch's matmuls (runtime check on ids, or accept the near-zero cost of an all-sentinel op with kernel support (a)).

### 3. Expert slot pool

Extends the prior proposal's layer-slot pool:

- Per device: a pool of expert-sized slots, `slot_bytes = nb[2]_gate + nb[2]_up + nb[2]_down` (per-layer sizes are uniform in practice; if not, size to max and note waste).
- Global VRAM budget `vram_budget_mb` → total slot count `K_total`; distributed per layer as `K_N` (see §5).
- Check-out/check-in, O(1), zero fragmentation across unlimited rebalance cycles.
- Slots are raw backend buffer ranges wrapped as the GPU partition pseudo-tensors via views; no per-rebalance allocation.

### 4. Rebalance execution: async copy + quiescent-point swap

Identical two-phase model as the prior proposal:

**Phase A — overlap with decode.** For each promotion, copy expert `i`'s three projection slices from the mmap'd master (`data + i*nb[2]`, contiguous) into checked-out slots on a side stream (`cudaMemcpyAsync` on a non-default stream; pinned staging buffer to get full PCIe bandwidth from pageable mmap memory). Demotions are metadata-only — the CPU master always holds every expert, so "moving to CPU" is just freeing the slot and flipping the remap entry.

**Phase B — decode-thread task at quiescent point.** After Phase A completes: update `remap` tensor contents (small `ggml_backend_tensor_set`, `n_expert × 4` bytes per layer), update `gpu_slot_of_expert` / `expert_of_gpu_slot`, return freed slots. Microseconds. No locking beyond the existing server task queue; no mid-graph state change; KV cache untouched.

**Cost model.** One expert (three projections) in a mid-size MoE is O(10–200 MB) depending on quant and n_ff. At ~25 GB/s effective PCIe 4.0 x16, promoting a dozen experts is tens of ms of *background* copy per rebalance. Rebalancing every few hundred tokens is comfortably cheap.

### 5. Rebalancing policy

- **Counters.** Router top-k IDs already reach the host each decode step; increment `uint32 counts[n_layer][n_expert]`. Maintain an EWMA or fixed sliding window (`window_tokens`) — aggregate lifetime counts converge toward uniform on load-balanced models and must not be used; *conditional* (session/topic) skew is the exploitable signal.
- **Per-layer budgets.** Allocate `K_N` proportional to measured per-layer skew (e.g., concentration of the window distribution), not uniformly — flat-routing layers waste slots. Recompute budgets on a slower cadence than expert rebalances since resizing partitions costs a re-reserve.
- **Hysteresis.** Promote when an expert's rank rises above `K_N − h`; demote only when it falls below `K_N + h` (`hysteresis: h`), and require a minimum usage delta. Prevents thrashing of experts oscillating around the boundary.
- **Trigger.** Every `interval_tokens` and/or at request boundaries (routing distribution shifts with topic; request boundaries are where it shifts). Manual mode disables the policy and honors only explicit POSTs.

### 6. HTTP API

```
GET  /v1/model/expert-placement            -> per-layer: K_N, resident expert ids, hit-rate stats
POST /v1/model/expert-placement            -> explicit placement (manual mode / external predictor)
GET  /v1/model/expert-placement/stats      -> windowed per-layer, per-expert usage counts
POST /v1/model/expert-placement/policy     -> auto/manual, interval, window, hysteresis, budget
```

Explicit POST expresses **the full desired GPU set per layer** (declarative, not deltas) — the server nets it against current state, so an external ML predictor can simply publish its target every interval. Handlers enqueue a rebalance task and await the result; per-request atomicity and validation errors as in the prior proposal. `GET .../stats` is the feed for training/driving an external residency predictor.

## Files Requiring Changes

| File | Changes |
|------|---------|
| `src/llama-model.h/.cpp` | `use_dynamic_experts` flag; expert partition state; expert slot pool; rebalance task (validate / net-out / Phase A copy / Phase B swap); keep loader mmap alive; per-expert slice offsets. |
| `src/llama-graph.cpp` | `build_moe_ffn`: dual-branch construction with remap gather, sentinel IDs, weighted merge under the flag. |
| `ggml/src/ggml-cuda/*` (mul_mat_id path) | Sentinel-skip support in MUL_MAT_ID (option (a)). |
| `ggml/src/ggml-cpu/*` | Same sentinel-skip for the CPU branch. |
| `src/llama-context.h/.cpp` | No re-reserve on remap-content updates; re-reserve hook for partition resizes only. |
| `tools/server/server-context.*`, `server.cpp` | Routing-counter collection in decode loop; rebalancer policy loop; four routes above; rebalance as decode-loop task. |

## Risks and Open Questions

1. **Kernel changes per backend.** Sentinel-skip in MUL_MAT_ID is the invasive part; scope v1 to CUDA + CPU, gate the feature on backend support, fall back to whole-tensor placement elsewhere.
2. **Flat routing = low ceiling.** If windowed top-`K_N` coverage is poor, GPU residency of experts buys little. Validation gates on measured coverage (below).
3. **CPU baseline may be strong.** On high-bandwidth many-core hosts with a small GPU, CPU expert matvec + activation round-trip may be near GPU speed; the mechanism must prove per-activation Δt on target hardware before the rebalancer matters.
4. **Repack/AMX CPU buffers.** As in the prior proposal, layout-transforming CPU buffer types are incompatible with mmap-as-master (and with global-ID indexing of raw GGUF bytes); disabled under the flag.
5. **Batch>1 / prefill.** Split execution is decode-oriented. For large prompt batches the existing per-op offload path is the right tool; the two branches remain correct but the GPU-partition benefit shrinks. Consider bypass (single CPU-branch execution) above a batch-size threshold.
6. **Interaction with speculative decoding / multiple sequences.** Counters aggregate across sequences; a shared working set is assumed. Per-sequence partitions are out of scope.
7. **gate/up/down coupling.** Moving projections together triples slot size but avoids cross-device activation hops; revisit only if profiling shows down-projection-only residency is worthwhile (unlikely).

## Implementation Phases

### Phase 0: Measurement (gates everything)
- Log router IDs on representative workloads; compute windowed top-K coverage per layer for candidate VRAM budgets.
- Microbenchmark per-expert matvec CPU vs GPU (+ round-trip) on target hardware.
- Go/no-go: projected speedup `coverage × Δt` must be material.

### Phase 1: Partition state + slot pool
- Expert slot pool; partition metadata; mmap slice offsets; manual placement applied at load time only (static per-expert placement — already a new capability vs `-ot`).

### Phase 2: Split-execution graph (masking bring-up)
- Dual-branch `build_moe_ffn` with weight-masking (option (b)); bit-exactness test vs baseline under greedy sampling with a fixed placement.

### Phase 3: Sentinel-skip kernels
- CUDA + CPU MUL_MAT_ID sentinel support; replace masking; perf test vs Phase 2 and vs `--n-cpu-moe` baseline.

### Phase 4: Runtime rebalance
- Phase A/B rebalance task; remap-content updates without re-reserve; 1000-cycle rebalance soak (no VRAM growth, no output divergence, decode-latency histogram during background copies).

### Phase 5: Counters + auto policy
- Windowed counters, per-layer budgets, hysteresis; end-to-end throughput A/B on real sessions vs static `--n-cpu-moe`.

### Phase 6: HTTP API
- Four endpoints; declarative placement; stats feed for external predictors.

## Alternatives Considered

1. **LRU expert cache with on-demand miss fill** (issue #20757 / discussion #23324 style). Better when workloads shift faster than any rebalance interval; pays complexity in eviction, admission, and miss stalls in the decode path. The partition scheme's remap indirection is a subset of the cache design, so this remains a compatible future extension — the graph work in Phases 2–3 is shared.
2. **Per-token prefetch.** Rejected: expert IDs are produced mid-graph with no lookahead; PCIe transfer latency per expert exceeds batch=1 compute by orders of magnitude. Only viable with speculative routing prediction, out of scope.
3. **Un-packing experts into `n_expert` separate tensors** and reusing the tensor-granular migration primitive. Rejected: loses MUL_MAT_ID fusion (per-expert mul_mat + gather is drastically slower), explodes tensor count and scheduler graph size, and still requires graph changes.
4. **Whole-tensor expert placement** (prior proposal / `--n-cpu-moe`). Remains the fallback for unsupported backends and the baseline for all benchmarks.

## Conclusion

Per-expert GPU placement cannot be expressed as tensor migration because GGUF packs all experts into one tensor per projection; it requires a dual-partition representation and a split MUL_MAT_ID execution path with a device-resident remap table. Everything else — mmap master copy, slot pooling, async copy with quiescent-point commit, batched declarative API — carries over from the tensor-granular proposal unchanged. Because the graph topology is static and only remap contents change, interval rebalancing costs background PCIe copies plus a microsecond metadata swap, with **zero scheduler re-reserves** in steady state. The design is gated on Phase 0 measurement: windowed expert-coverage and CPU/GPU matvec deltas on target hardware determine whether the mechanism pays for itself before any kernel work begins.