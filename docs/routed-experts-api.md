# `/v1/moe/routed-experts` API

This endpoint streams MoE (Mixture of Experts) router decisions as they happen during inference.

## Endpoint

```text
GET /v1/moe/routed-experts
```

Returns a streaming response (`text/event-stream`) containing one Server-Sent Event (SSE) per batch of routed tokens.

## What it returns

Each event contains a JSON object with the following fields:

| Field      | Type             | Description                                                             |
|------------|------------------|-------------------------------------------------------------------------|
| `layer`    | integer          | Transformer layer index that produced the routing decision.             |
| `token_id` | integer          | The actual tokenizer token ID that was routed through that layer.       |
| `experts`  | array of integers| Selected expert indices, ordered by router score (highest score first). |
| `backend`  | object           | Residency (`cpu` or `gpu`) of each expert weight tensor in the layer.   |

The `backend` object has one entry for each possible expert weight tensor:

| Key                 | Value  | Tensor checked                                   |
|---------------------|--------|--------------------------------------------------|
| `ffn_down_exps`     | `cpu` or `gpu` | `blk.<layer>.ffn_down_exps.weight`      |
| `ffn_gate_up_exps`  | `cpu` or `gpu` | `blk.<layer>.ffn_gate_up_exps.weight`   |
| `ffn_up_exps`       | `cpu` or `gpu` | `blk.<layer>.ffn_up_exps.weight`        |
| `ffn_gate_exps`     | `cpu` or `gpu` | `blk.<layer>.ffn_gate_exps.weight`      |

If a tensor does not exist for the current architecture, its value defaults to `cpu`.

Example stream:

```text
data: {"layer":15,"token_id":15,"experts":[52,8,58,30,60,4,16,17],"backend":{"ffn_down_exps":"gpu","ffn_gate_up_exps":"gpu","ffn_up_exps":"cpu","ffn_gate_exps":"cpu"}}

data: {"layer":16,"token_id":15,"experts":[14,46,43,30,58,38,61,4],"backend":{"ffn_down_exps":"gpu","ffn_gate_up_exps":"gpu","ffn_up_exps":"cpu","ffn_gate_exps":"cpu"}}

data: {"layer":17,"token_id":15,"experts":[54,3,53,56,8,23,0,42],"backend":{"ffn_down_exps":"gpu","ffn_gate_up_exps":"gpu","ffn_up_exps":"cpu","ffn_gate_exps":"cpu"}}
```

For MoE models such as `allenai/OLMoE-1B-7B-0924`, the `experts` array typically contains 8 indices drawn from a pool of 64 experts.

## How it works

### 1. Hook into graph execution

The implementation registers a `ggml_backend_sched_eval_callback` when the server's `llama_context` is created. The callback is set in `tools/server/server-context.cpp`:

```cpp
params_base.cb_eval           = server_moe_router_cb;
params_base.cb_eval_user_data = &moe_router_state;
```

### 2. Capture the current batch tokens

Right before each `llama_decode()` call, the token IDs in the current batch view are copied into `moe_router_state.tokens`:

```cpp
moe_router_state.tokens.assign(batch_view.token, batch_view.token + batch_view.n_tokens);
```

This lets the callback map the tensor's batch position back to an actual `llama_token` ID.

### 3. Identify the routing tensor

During graph execution, the callback is invoked for every computed tensor. In the `ask` phase it returns `true` only for tensors whose name starts with `ffn_moe_topk`. These tensors are produced by the router in `llm_graph_context::build_moe_ffn()`:

```cpp
ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used);
cb(selected_experts, "ffn_moe_topk", il);
```

The tensor shape is `[n_expert_used, n_tokens]` and its type is `GGML_TYPE_I32`.

### 4. Build samples

When the callback observes a matching tensor, it:

- Parses the layer index from the tensor name (`ffn_moe_topk-<layer>`).
- Reads the selected expert indices from GPU/CPU memory using `ggml_backend_tensor_get()`.
- For each token in the tensor, creates a `server_moe_router_sample` containing `layer`, `token_id`, and `experts`.
- Pushes the sample into a thread-safe queue (`moe_router_state.samples`).

The queue is bounded at 10,000 samples; old samples are dropped when the queue is full and no client is reading.

### 5. Stream to the client

The HTTP handler for `GET /v1/moe/routed-experts` returns a streaming `server_http_res`. Its `next` function drains the sample queue, batches up to 128 samples, and formats each batch as SSE events:

```cpp
chunk += "data: " + j.dump() + "\n\n";
```

If no samples are available, the handler waits up to 100 ms, checks whether the client has disconnected, and otherwise returns an empty chunk to keep the connection alive.

## Files changed

| File | Change |
|------|--------|
| `tools/server/server-context.cpp` | Added `server_moe_router_sample`, `server_moe_router_state`, the `server_moe_router_cb` callback, batch-token capture before `llama_decode()`, and the `get_moe_routed_experts` handler. |
| `tools/server/server-context.h` | Declared the `get_moe_routed_experts` handler member. |
| `tools/server/server.cpp` | Registered the `GET /v1/moe/routed-experts` route. |

## Console logging

The same callback also continues to print routing decisions to the console:

```text
[moe-router] ffn_moe_topk-15 token  0 (id     15): 52 8 58 30 60 4 16 17
```

This is independent of the HTTP stream and can be removed or gated later if the console noise is unwanted.

## Usage

Start `llama-server` with an MoE model, then run:

```bash
curl -N [REDACTED-URL]
```

While the stream is open, send a normal chat completion request. The stream will emit one or more events for every layer that routes tokens during that request.

## Limitations

- The callback runs synchronously inside `llama_decode()`, so very heavy per-sample work could slow down generation. The current implementation only copies small I32 tensors and pushes them to a queue.
- Samples are global across all slots/sequences. There is no per-slot filtering yet.
- The queue drops old samples if no client is connected, so starting the stream after a request has already generated tokens will not show historical data.
