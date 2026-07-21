# Plan Review v2: Device-Resident MoE Expert Stats

## Verdict: APPROVED

All six technical questions verify correctly against the ggml source. The revised `ggml_get_rows_back` + `ggml_acc_inplace` approach is feasible. One minor caveat noted under Findings #4 (the acc stride values are oversized but harmless for a 1D tensor).

## Findings

### 1. ggml_get_rows_back assertions

**PASS.** Source: `ggml/src/ggml.c:3882-3883`:
```c
GGML_ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b->type == GGML_TYPE_I32);
GGML_ASSERT(ggml_is_matrix(c) && (a->ne[0] == c->ne[0]));
```
- `a = ones [1, n_sel]` F32 2D: `ne[2]==1 && ne[3]==1` → `ggml_is_matrix` true. `ne[0]=1`.
- `b = flat_ids [n_sel]` I32 1D (from `ggml_view_1d`): `ne[1]==ne[2]==ne[3]==1` → `ggml_is_vector` true. Type I32.
- `c = shape_ref [1, n_expert]` F32 2D (from `ggml_view_2d`): `ggml_is_matrix` true. `ne[0]=1`.
- `a->ne[0] (1) == c->ne[0] (1)` ✓.
- Output: `ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c->ne[0], c->ne[1])` = `[1, n_expert]` F32. ✓
- Return type is hardcoded F32 regardless of input types (line 3886). ✓

### 2. k_get_rows_back_float scatter-add semantics

**PASS — scatter-ADD (sum), not scatter-SET.** Source: `ggml/src/ggml-cuda/getrows.cu:80-95`:
```cpp
for (int64_t i = 0; i < nrows_grad; ++i) {
    if (rows[i] != dst_row) { continue; }
    sum += grad[i*ncols + col];
}
dst[dst_row*ncols + col] = sum;
```
- `nrows_grad = ne10 = src1->ne[0] = n_sel` (the flattened index count).
- `dst_row` ranges over `[0, ne1) = [0, n_expert)` (block_nums.y = ne1, line 307).
- With `grad = ones` (all 1.0) and `rows = flat_ids`, `dst[e] = count of i where flat_ids[i] == e`. ✓ This is a histogram.
- CUDA op (`ggml_cuda_op_get_rows_back`, line 279) asserts F32 src0/dst, I32 src1, all three contiguous, and `ne02*ne03==1`, `ne12*ne13==1`, `ne2*ne3==1` — all satisfied by the plan's shapes (a is `[1, n_sel, 1, 1]`, b is `[n_sel, 1, 1, 1]`, dst is `[1, n_expert, 1, 1]`).

### 3. ggml_view_2d for shape_ref

**PASS.** Source: `ggml/src/ggml.c:3710-3725`. `ggml_view_2d(ctx, a, ne0, ne1, nb1, offset)` creates a 2D tensor with `ne=[ne0, ne1]`, `nb[1]=nb1`, `nb[2]=nb[1]*ne1`, `nb[3]=nb[2]`.
- `moe_stats_buf` is `[n_expert, n_layer]` F32, contiguous, `nbytes = n_expert * n_layer * 4`.
- `shape_ref = ggml_view_2d(ctx0, moe_stats_buf, 1, n_expert, sizeof(float), 0)`:
  - `ne = [1, n_expert]`, `nb[1] = 4`, `nb[2] = 4 * n_expert`, `nb[3] = 4 * n_expert`.
  - `data_size = 4 * 1 * n_expert = 4 * n_expert`; `view_offs = 0`; `0 + 4*n_expert <= nbytes` ✓. Valid view.
  - Produces `[1, n_expert]` 2D F32. Only the shape is consumed by `ggml_get_rows_back` (output is a fresh tensor, `c`'s data is never read). ✓
- Note: `shape_ref` aliases `moe_stats_buf`'s data, but since `get_rows_back` only reads `c->ne[]` and writes to a fresh `result` tensor, there is no aliasing hazard with the subsequent `acc` into `moe_stats_buf`.

### 4. ggml_acc_inplace on 1D views

**PASS (with caveat).** Sources: `ggml/src/ggml.c:2140-2155` (assertions) and `ggml/src/ggml-cuda/acc.cu:37-67` (kernel).
- `ggml_acc_impl` asserts: `ggml_nelements(b) <= ggml_nelements(a)` (both `n_expert` ✓), `ggml_is_contiguous(a)` ✓ (view_1d is contiguous — `nb[0]=type_size`, `ne[1..3]=1`), `a->type == F32` ✓, `b->type == F32` ✓ (get_rows_back returns F32).
- CUDA op (`ggml-cuda/acc.cu:37`) additionally asserts: `src0->type==F32`, `src1->type==F32`, `dst->type==F32`, `ggml_is_contiguous(src1)`, `dst->nb[0] == ggml_element_size(dst)`, `ggml_is_contiguously_allocated(dst)`. For `stats_view` (view_1d of a contiguous device buffer at byte offset `il*n_expert*4`): `nb[0]=4=element_size` ✓, allocation is contiguous ✓.
- Stride params: plan passes `nb1=nb2=nb3=n_expert*sizeof(float)`. Kernel reads `s1=s2=s3=n_expert` (after `/sizeof(float)`). For 1D `b` (`ne[0]=n_expert`, `ne[1..3]=1`): the index decomposition in `acc_f32` (lines 12-23) yields `i13=i/s3=0`, `i12=0`, `i11=i/s1` (0 when `i<n_expert`), `i10=i`. So `val += y[i]` for `i<n_expert`. Correct add. The oversized `s2/s3` are harmless because `ne12=ne13=1` bounds the index check. ✓
- Caveat (not a bug): `n_expert*sizeof(float)` for `nb2/nb3` is technically the row stride of `b` viewed as 2D/3D, but since `b` is 1D these never gate the read. Passing `n_expert*sizeof(float)` for all three is functionally correct; a stricter author might pass `n_expert*sizeof(float)` for nb1 and `0` (or any value) for nb2/nb3, but the current choice works.

### 5. Multi-token correctness

**PASS.** With `n_tokens > 1`, `selected_experts` is `[n_expert_used, n_tokens]` I32. `flat_ids = ggml_view_1d(ctx0, selected_experts, n_sel, 0)` with `n_sel = n_expert_used * n_tokens` flattens row-major (ggml is column-major-contiguous, and `[n_expert_used, n_tokens]` has `nb[0]=4`, `nb[1]=n_expert_used*4`; a 1D view of length `n_sel` reads `n_sel` consecutive I32 elements = the entire tensor — correct flattening). The scatter-add kernel iterates all `n_sel` indices and increments `dst[e]` for every `flat_ids[i]==e`, so counts aggregate across all tokens in the batch. ✓

### 6. selected_experts shape

**PASS.** Source: `src/llama-graph.cpp:1605`:
```cpp
ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
```
Comment confirms `[n_expert_used, n_tokens]` I32 (argsort_top_k preserves input dtype; `selection_probs` is F32 but argsort returns I32 indices). Downstream `ggml_get_rows(ctx0, probs, selected_experts)` at line 1618 also treats it as `[n_expert_used, n_tokens]`. ✓

## Issues

None. The revised plan is technically sound. Minor non-blocking note: in Finding #4, the `nb2`/`nb3` stride arguments to `ggml_acc_inplace` are oversized for a 1D tensor but produce correct results because the kernel's `ne12`/`ne13` bounds (both 1) prevent any out-of-range read.
