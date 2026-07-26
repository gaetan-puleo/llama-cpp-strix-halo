# Inference Path on ROCm

This is a linear map of ordinary decoder-only inference in llama.cpp on an AMD
GPU. It uses the small program in `examples/simple/simple.cpp` and the Llama
model graph as concrete examples.

The exact transformer graph changes with the model architecture. The control
flow from `llama_decode()` through the scheduler to the HIP backend stays the
same.

## One-screen map

```text
text
  -> tokens
  -> llama_batch
  -> llama_decode()
  -> micro-batch + KV-cache slot
  -> model-specific ggml graph
  -> backend scheduler
  -> HIP backend graph execution
  -> HIP kernels / hipBLAS
  -> logits copied to host
  -> sampler chooses one token
  -> token becomes the next one-token batch
  -> repeat
```

Keep this distinction in mind:

- llama.cpp controls the model, batches, KV cache, and generation loop.
- GGML describes tensor operations and schedules them.
- The HIP backend turns scheduled operations into GPU work.
- ROCm executes that work on the Strix Halo GPU.

## The important objects

| Object | Meaning | Lifetime |
| --- | --- | --- |
| `llama_model` | Metadata, vocabulary, weights, and device placement | Whole run |
| `llama_context` | One inference state: backends, scheduler, KV cache, outputs | Whole conversation |
| `llama_batch` | Tokens submitted by the caller | One `llama_decode()` call |
| `llama_ubatch` | Internal micro-batch | Part of one decode call |
| `ggml_tensor` | Tensor metadata plus an operation and its inputs | One graph or persistent weight/cache |
| `ggml_cgraph` | Ordered tensor-operation graph | One graph shape, often reused |
| `ggml_backend_sched` | Assigns graph sections and copies to backends | Context lifetime |
| KV cache | K and V states from earlier positions | Context lifetime |
| logits | One score per vocabulary token | Decode output |

## 1. Load backend modules

The minimal example starts at `examples/simple/simple.cpp:80`:

```cpp
ggml_backend_load_all();
```

`ggml_backend_load_all()` tries backend modules including `hip` and `cpu` in
`ggml/src/ggml-backend-reg.cpp:562`. A ROCm build provides the `ggml-hip`
module.

The HIP build is less separate from CUDA than its name suggests:

- `ggml/src/ggml-hip/CMakeLists.txt:60` collects headers from `ggml-cuda`.
- `ggml/src/ggml-hip/CMakeLists.txt:63` compiles the `.cu` sources from
  `ggml-cuda` as HIP.
- `GGML_USE_HIP` selects `ggml/src/ggml-cuda/vendors/hip.h`.
- `vendors/hip.h` maps names such as `cudaMemcpyAsync` and `cublasGemmEx` to
  `hipMemcpyAsync` and `hipblasGemmEx`.

Therefore, names containing `cuda` in this backend usually name shared source
code. On a ROCm build, they compile to HIP operations, not CUDA runtime calls.

**Checkpoint:** the HIP module is now registered. No model has run yet.

## 2. Load the GGUF model

The caller creates `llama_model_params`, sets `n_gpu_layers`, and calls
`llama_model_load_from_file()` at `examples/simple/simple.cpp:84`.

Loading performs four relevant jobs:

1. Read GGUF metadata and vocabulary.
2. Create the architecture-specific `llama_model` implementation.
3. Create tensor descriptors for every weight.
4. Allocate buffers and load weight bytes into them.

Device placement happens in `llama_model_base::load_tensors()` at
`src/llama-model.cpp:1244`:

- The input layer stays on CPU (`src/llama-model.cpp:1320`).
- Repeating transformer layers are assigned according to `n_gpu_layers` and
  the device split (`src/llama-model.cpp:1306`).
- The output layer is assigned after the repeating layers
  (`src/llama-model.cpp:1330`).
- Weight buffers are marked `GGML_BACKEND_BUFFER_USAGE_WEIGHTS`. The scheduler
  uses this to prefer the backend that already owns a weight
  (`src/llama-model.cpp:1603`).

For a single Strix Halo GPU with all layers offloaded, nearly all large matrix
weights are in a HIP backend buffer. "All layers" still does not mean that all
program logic runs on the GPU.

Strix Halo is an integrated GPU: CPU and GPU ultimately use the same physical
DRAM pool. The software still has distinct host/device buffer types, ownership,
HIP streams, synchronization, and copy operations. Shared physical memory can
make transfers cheaper; it does not remove these abstractions.

**Checkpoint:** weights have fixed homes. They are not reloaded for every token.

## 3. Tokenize the prompt

`llama_tokenize()` at `examples/simple/simple.cpp:97` converts UTF-8 text into
token IDs using the vocabulary stored in the GGUF.

The transformer never sees text. It receives integer token IDs. Conversion
back to text later uses `llama_token_to_piece()`.

**Checkpoint:** prompt text is now a vector of integers.

## 4. Create the inference context

`llama_init_from_model()` at `examples/simple/simple.cpp:109` constructs a
`llama_context`.

Its constructor in `src/llama-context.cpp:82`:

1. Resolves context size, batch size, micro-batch size, RoPE, Flash Attention,
   and other runtime parameters.
2. Initializes one backend for each selected GPU device
   (`src/llama-context.cpp:325`).
3. Adds the CPU backend last (`src/llama-context.cpp:347`). CPU remains the
   fallback for unsupported or deliberately host-side operations.
4. Creates the model's memory module, normally the KV cache
   (`src/llama-context.cpp:380`).
5. Creates a `ggml_backend_sched` and reserves worst-case prompt-processing and
   token-generation graphs (`src/llama-context.cpp:573`).

The scheduler's temporary compute buffers are separate from:

- persistent model weight buffers;
- persistent KV-cache buffers;
- the host-visible output buffer.

Reservation avoids allocating all intermediate tensors again for each token.

**Checkpoint:** the model is immutable; the context contains mutable inference
state.

## 5. Build the caller's batch

The first batch contains every prompt token:

```cpp
llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
```

This is prompt processing, also called **prefill**. Later batches normally
contain one sampled token. That phase is token generation or **decode**.

The same `llama_decode()` API handles both. The tensor shapes differ, so the
backend can choose different kernels:

- Prefill has many columns and resembles matrix-matrix work.
- Generation usually has one column and resembles matrix-vector work.

**Checkpoint:** prefill and generation use the same path but different shapes.

## 6. Enter `llama_decode()`

The public C API at `src/llama-context.cpp:4071` calls
`llama_context::decode()` at `src/llama-context.cpp:1693`.

`decode()` proceeds in this order:

1. Validate the batch.
2. Normalize missing positions, sequence IDs, and output flags through the
   batch allocator.
3. Ensure the batch does not exceed `n_batch`.
4. Apply pending KV-cache maintenance.
5. Ask the memory module for cache slots and a micro-batch.
6. Reserve host-visible output space.
7. Process each `llama_ubatch`.

`n_batch` is the maximum caller batch. `n_ubatch` is the maximum internal piece
processed by one graph execution. A large prompt may therefore require several
graph executions.

The memory module assigns each token position to KV-cache cells before the
graph runs. If no suitable cells exist, decode may optimize the cache and retry
(`src/llama-context.cpp:1778`).

**Checkpoint:** each micro-batch now has positions, sequence IDs, output rows,
and KV-cache locations.

## 7. Build or reuse the GGML graph

`llama_context::process_ubatch()` is at `src/llama-context.cpp:1317`.

It computes graph parameters from the micro-batch. If the full topology matches
the previous run, llama.cpp reuses the graph. Otherwise it:

1. Resets the old graph result and scheduler state.
2. Calls `llama_model::build_graph()` (`src/llama-model.cpp:2290`).
3. Lets the scheduler allocate graph intermediates.

Graph reuse reuses topology and allocations, not numerical results. Input
tensors are filled again before every execution at
`src/llama-context.cpp:1367`.

A `ggml_tensor` can represent data or an unevaluated operation. For example,
`ggml_mul_mat()` creates an output tensor whose operation is
`GGML_OP_MUL_MAT`; it does not immediately multiply matrices.

**Checkpoint:** the graph says what to compute, not yet how ROCm computes it.

## 8. Construct one Llama transformer graph

Architecture dispatch occurs in `llama_model::build_graph()`. For Llama,
`llama_model_llama::build_arch_graph()` enters `src/models/llama.cpp:94`.

The graph is constructed in model order:

1. Look up token embeddings.
2. Create position and attention-mask inputs.
3. For every transformer layer:
   - RMS-normalize the layer input.
   - Multiply by Q, K, and V projection weights.
   - apply RoPE to Q and K;
   - write new K and V values into the KV cache;
   - attend Q over cached K and V;
   - apply the attention output projection;
   - add the attention residual;
   - RMS-normalize for the feed-forward network;
   - compute gate and up projections, SiLU gating, and down projection;
   - add the feed-forward residual.
4. Apply final RMS normalization.
5. Apply the language-model head to produce logits.
6. Expand dependencies into an ordered `ggml_cgraph`.

The concrete Llama layer loop is `src/models/llama.cpp:126`. The final head and
graph expansion are at `src/models/llama.cpp:229`.

### What is the FFN gate?

A gate controls how much of each intermediate feature passes through the
feed-forward network. It is continuous, not simply open or closed.

```text
input x
  |-> up   = W_up x ------------------|
  |                                    | elementwise multiply
  |-> gate = SiLU(W_gate x) ----------|
                                       |
                         output = W_down(up * gate)
```

For each feature:

- a gate value near zero suppresses it;
- a larger positive value passes or amplifies it;
- a negative value can invert or suppress it.

The up projection proposes features. The gate projection decides how much of
each proposed feature survives. The down projection returns the expanded FFN
representation to the model's hidden size.

In `src/models/llama.cpp:189`, `build_ffn()` receives `ffn_up`, `ffn_gate`, and
`ffn_down`, with `LLM_FFN_SILU`. Their weights are loaded at
`src/models/llama.cpp:69`.

Attention has two important forms in `src/llama-graph.cpp:2407`:

- Flash Attention: one `GGML_OP_FLASH_ATTN_EXT` operation combines the main
  attention work.
- Non-flash: separate K-Q multiplication, masking, softmax, and V multiplication
  operations appear in the graph.

The KV cache prevents recomputing K and V for all old tokens. During generation,
the new token contributes one K row and one V row per layer; its Q still attends
over the relevant previous cache entries.

**Checkpoint:** the transformer is now a dependency graph of generic GGML ops.

## 9. Assign graph nodes to backends

The graph is submitted by `llama_context::graph_compute()` at
`src/llama-context.cpp:2438`:

```cpp
ggml_backend_sched_graph_compute_async(sched.get(), gf);
```

Before execution, the scheduler:

1. Assigns each node to a backend that supports its operation.
2. Prefers the backend containing the node's weights.
3. Groups adjacent nodes for the same backend into graph splits.
4. Inserts cross-backend tensor copies.
5. Allocates intermediate tensors from reusable compute buffers.

This logic is in `ggml/src/ggml-backend.cpp`. Split execution starts at line
1541; each split is sent to `ggml_backend_graph_compute_async()` at line 1678.

With full GPU offload, most transformer nodes form HIP splits. CPU splits still
can exist. Partial offload adds boundaries where activations must move between
CPU and HIP backend buffers.

Use `GGML_SCHED_DEBUG=1` to print backend assignments in a debug build.

**Checkpoint:** every operation now has an executor and every intermediate has
storage.

## 10. Enter the HIP backend

The HIP backend exposes the shared CUDA backend interface. Its graph callback is
`ggml_backend_cuda_graph_compute()` at
`ggml/src/ggml-cuda/ggml-cuda.cu:4126`.

For each GPU split it:

1. Selects the HIP device.
2. Optionally checks whether a reusable HIP graph can be captured or launched
   when HIP graphs were enabled at build time.
3. Walks graph nodes in dependency order.
4. Skips views and metadata-only operations.
5. Tries supported operation fusions.
6. Calls `ggml_cuda_compute_forward()` for each remaining node.

The operation dispatch switch starts at
`ggml/src/ggml-cuda/ggml-cuda.cu:2011`. Examples:

| GGML operation | HIP backend path |
| --- | --- |
| `GGML_OP_GET_ROWS` | embedding row lookup kernel |
| `GGML_OP_RMS_NORM` | RMS normalization kernel |
| `GGML_OP_MUL_MAT` | matrix multiplication selector |
| `GGML_OP_ROPE` | RoPE kernel |
| `GGML_OP_FLASH_ATTN_EXT` | selected Flash Attention kernel |
| `GGML_OP_ADD` | elementwise add kernel |

Calls still spelled `cuda...` are macro-mapped to HIP in
`ggml/src/ggml-cuda/vendors/hip.h`. Kernel launch syntax is compiled by the HIP
compiler for the selected AMDGPU target.

## 11. Select matrix multiplication kernels

Matrix multiplication dominates most dense transformer inference. The selector
is `ggml_cuda_mul_mat()` at `ggml/src/ggml-cuda/ggml-cuda.cu:1812`.

It chooses according to weight type, activation type, dimensions, and detected
GPU architecture:

1. `mmvf`: custom floating-point matrix-vector path.
2. `mmf`: custom floating-point matrix path.
3. `mmvq`: custom quantized matrix-vector path.
4. `mmq`: custom quantized matrix path.
5. hipBLAS fallback.

Small one-token generation and wide multi-token prefill can therefore use
different kernels for the same model weight.

On the hipBLAS path, source names such as `cublasGemmEx` map to
`hipblasGemmEx`. hipBLAS may then use rocBLAS internally. On custom quantized
paths, llama.cpp's own HIP-compiled kernels perform dequantization and dot
products without first expanding the entire weight matrix in memory.

Strix Halo is classified as RDNA 3.5 in `ggml/src/ggml-cuda/common.cuh:80`.
These architecture checks influence which paths are considered valid and fast.
The exact choice also depends on model quantization and batch shape, so there is
no single "Strix Halo kernel" for all inference.

**Checkpoint:** generic `GGML_OP_MUL_MAT` has become concrete HIP kernel or
hipBLAS work queued on a stream.

## 12. Finish asynchronously and expose logits

GPU work is normally queued asynchronously. After graph submission,
`llama_context::decode()` asks which backend owns the logits tensor and starts
an asynchronous copy to the context's host-visible output buffer at
`src/llama-context.cpp:1902`.

Synchronization is deferred until host code needs the output. Access through
`llama_get_logits_ith()` performs the required synchronization before returning
a usable pointer.

Usually only requested output rows are produced. During ordinary generation,
that is the logits row for the last input token, not one row for every token in
the context.

**Checkpoint:** the CPU can now read one score for every vocabulary token.

## 13. Sample one token

The example calls `llama_sampler_sample()` at
`examples/simple/simple.cpp:180`. Its implementation is
`src/llama-sampler.cpp:806`.

For normal host sampling it:

1. Gets the selected logits row.
2. Creates candidates `(token_id, logit, probability)`.
3. Applies the sampler chain in order, such as penalties, temperature, top-k,
   top-p, and distribution sampling.
4. Accepts and returns the selected token.

The minimal example uses greedy sampling, so it selects the highest logit.
Sampling is not part of the transformer forward pass unless backend sampling
was explicitly configured.

The token ID is converted to a byte piece for display. Token pieces are not
guaranteed to be complete UTF-8 characters individually; applications should
handle streaming bytes accordingly.

## 14. Repeat the generation loop

The sampled token becomes a new one-token `llama_batch` at
`examples/simple/simple.cpp:199`. The loop calls `llama_decode()` again.

On a stable one-token shape:

- weights stay allocated;
- KV-cache history stays allocated and grows logically;
- graph topology and compute allocations are usually reused;
- only new inputs and changing cache positions are filled;
- GPU kernels run for the new token;
- one logits row returns for sampling.

Generation stops on an end-of-generation token, a caller limit, an abort, or an
error.

## Data movement summary

```text
GGUF file
  -> model weight buffers                         once at load

prompt token IDs + positions + masks
  -> graph input buffers                          each micro-batch

new K and V
  -> persistent KV-cache buffers                  each layer, each token

intermediate activations
  -> reusable backend compute buffers             each graph execution

selected logits rows
  -> host-visible context output buffer           each decode with outputs

sampled token ID
  -> next caller batch                            each generation step
```

On Strix Halo these paths share physical DRAM bandwidth. Model weights, KV
cache, activations, CPU activity, and display use the same memory system. The
most useful performance question is often not only "how many FLOPs?" but also
"which bytes are read, written, converted, or copied for each token?"

## What the main options change

| Option or parameter | Main consequence |
| --- | --- |
| `-ngl`, `n_gpu_layers` | Weight placement and CPU/GPU split boundaries |
| `-c`, `n_ctx` | KV-cache capacity and memory use |
| `-b`, `n_batch` | Largest caller batch accepted |
| `-ub`, `n_ubatch` | Largest graph execution chunk |
| Flash Attention | Fused attention graph and usually lower temporary memory |
| KV cache K/V types | Cache size, bandwidth, supported attention paths |
| Model quantization | Weight size and matrix-kernel selection |
| Sampler settings | Token choice after logits; usually little transformer cost |

Do not confuse `n_batch` with context size. Context size is retained history;
batch size is work submitted in one call.

## A practical reading order

Read one level at a time and stop after each checkpoint:

1. `examples/simple/simple.cpp:80-204` - complete outer loop.
2. `src/llama-context.cpp:1693-2068` - one decode call.
3. `src/llama-context.cpp:1317-1387` - one micro-batch graph execution.
4. `src/models/llama.cpp:94-247` - one representative transformer graph.
5. `src/llama-graph.cpp:2380-2520` - attention construction.
6. `ggml/src/ggml-backend.cpp:1541-1725` - scheduler split execution.
7. `ggml/src/ggml-cuda/ggml-cuda.cu:2011-2244` - operation dispatch.
8. `ggml/src/ggml-cuda/ggml-cuda.cu:1812-1852` - matrix-kernel choice.
9. `ggml/src/ggml-cuda/vendors/hip.h` - CUDA-name to HIP mapping.
10. `src/llama-sampler.cpp:806-872` - logits to next token.

## Useful observations

Run a small model first so logs remain readable:

```sh
GGML_SCHED_DEBUG=1 ./build/bin/llama-simple -m model.gguf "Hello"
```

Useful ROCm tools and variables:

```sh
rocminfo
rocm-smi
AMD_LOG_LEVEL=3 ./build/bin/llama-simple -m model.gguf "Hello"
```

For timing, compare prompt processing (`prompt eval time`) with token generation
(`eval time`). They exercise different shapes and often different matrix
kernels.

To inspect a graph, temporarily enable the existing `ggml_graph_dump_dot()`
debug block near `src/llama-context.cpp:1889`, then render the DOT output. Treat
this as a debugging edit, not a normal runtime option.

## Compact glossary

- **Backend:** executor and memory implementation for CPU, HIP, or another
  device API.
- **Decode:** in llama.cpp, evaluate a batch with a decoder model. It includes
  both prompt prefill and one-token generation.
- **Graph:** tensor operations and dependencies for one forward pass shape.
- **Logit:** unnormalized score for one vocabulary token.
- **Prefill:** process prompt tokens, usually in parallel batches.
- **Generation:** process newly sampled tokens, usually one per sequence.
- **KV cache:** retained attention keys and values from previous positions.
- **Micro-batch:** internal subdivision of the caller's batch.
- **Offload:** place weights or operations on a non-CPU backend.
- **Quantization:** compact weight representation with specialized dot-product
  kernels.
- **RoPE:** position-dependent rotation applied to attention Q and K.
- **Split:** consecutive graph section assigned to one backend.
- **Stream:** ordered asynchronous HIP work queue.

## Final mental model

One generated token is not one GPU kernel. It is one forward graph containing
many operations across every layer. The scheduler sends its HIP-supported
sections to the GPU. The HIP backend selects and queues many kernels, returns
one logits row, and the sampler chooses the integer that starts the next pass.
