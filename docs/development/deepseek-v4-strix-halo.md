# DeepSeek V4 Flash on Strix Halo

## Scope and status

This branch implements the native DeepSeek V4 Flash sparse-attention architecture for the Radeon 8060S (`gfx1151`) in llama.cpp. It is specialized for the local DeepSeek V4 Flash 0731 IQ3_XXS GGUF and favors model quality over stock-path numerical identity.

The model is deployed through llama-swap as:

```text
API base: http://100.106.243.98:8080/v1
Model ID: deepseek-v4-flash:origin
```

The older Vulkan model remains available as `deepseek-v4-flash:iq3`.

The DeepSeek architecture and the Strix Halo execution path are integrated and operational. CSA passes raw and compressed K=V cache regions directly to sparse attention and no longer materializes `concat(raw_k, csa_k)`. The visibility masks are still concatenated; this copy is smaller than the removed D512 cache copy.

## Model inventory

Model:

```text
/home/gaetan/models/deepseek/deepseek-v4-flash-0731/unsloth/
DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf
```

The model has four shards, 1,328 tensors, 284.33B parameters, and a reported size of 97.05 GiB.

### Architecture metadata

| Key | Value |
| --- | ---: |
| `general.architecture` | `deepseek4` |
| `deepseek4.block_count` | 43 |
| `deepseek4.context_length` | 1,048,576 |
| `deepseek4.embedding_length` | 4,096 |
| Query heads | 64 |
| KV heads | 1 |
| Main Q/K/V head size | 512 |
| RoPE dimensions | 64 |
| Q LoRA rank | 1,024 |
| Sliding window | 128 |
| Indexer heads | 64 |
| Indexer head size | 128 |
| Indexer Top-K | 512 |
| Output groups | 8 |
| Output LoRA rank | 1,024 |
| Routed experts | 256 |
| Active experts | 6 |
| Expert FFN size | 2,048 |
| Hyper-connection streams | 4 |
| Sinkhorn iterations | 20 |
| Compressed RoPE base | 160,000 |

The compression-ratio metadata begins with:

```text
[0, 0, 4, 128, 4, 128, ..., 4]
```

It is followed by three ratio-zero auxiliary entries. The 43 main transformer layers are the first 43 entries.

### Tensor types

| GGUF type | Tensors | Elements |
| --- | ---: | ---: |
| BF16 | 43 | 45,088,768 |
| F32 | 662 | 41,266,775 |
| I32 | 3 | 2,327,040 |
| IQ2_XS | 50 | 107,374,182,400 |
| IQ3_S | 2 | 4,294,967,296 |
| IQ3_XXS | 75 | 161,061,273,600 |
| MXFP4 | 2 | 4,294,967,296 |
| Q6_K | 170 | 2,292,187,136 |
| Q8_0 | 321 | 4,928,307,200 |

The two MXFP4 tensors in this GGUF are routed-expert FFN weights, not Lightning Indexer weights. The Indexer uses Q8_0 projections, F32 head weights and APE values, and F32 normalization weights. Main attention projections are predominantly Q8_0, with Q6_K used for most `attn_q_a` tensors.

## Official DeepSeek architecture

The verified Flash layer schedule is:

| Layers | Mode | Compressed branch | Local branch |
| --- | --- | --- | --- |
| 0-1 | Pure SWA | None | 128 tokens |
| Even 2-42 | CSA | Ratio 4, Top-512 | 128 tokens |
| Odd 3-41 | HCA | Ratio 128, all completed rows | 128 tokens |

This gives 2 pure-SWA layers, 21 CSA layers, and 20 HCA layers.

For query position `t`, the local branch contains at most 128 tokens and includes the current token:

```text
[max(0, t - 127), ..., t]
```

CSA exposes `floor((t + 1) / 4)` completed rows and selects at most 512. HCA exposes every completed ratio-128 row. At the 1M context limit, CSA attends to at most 128 + 512 rows, while HCA attends to 128 + 8,192 rows.

CSA compression is an overlapping learned pool. It combines four projected values from the current ratio-4 block and four from the previous block using a per-dimension softmax. HCA pools one non-overlapping group of 128 projected values. Both normalize the compressed row and apply partial RoPE after pooling.

The main attention is MQA. All 64 query heads share one 512-dimensional cache row, and the same row is used as both K and V. Each head has an attention sink. The last 64 output dimensions receive inverse query-position RoPE before the grouped output projection.

The public reference constructs sparse IDs as:

```text
[128 local IDs in chronological order][up to 512 CSA IDs in score order]
```

`torch.topk` defaults to descending-score order. The paper defines a selected set and does not require a production ordering, but this branch preserves the public-reference ordering.

### Lightning Indexer

The score for compressed row `s` is:

```text
sum_h weight[t,h] * ReLU(dot(index_q[t,h], index_k[s]))
```

The total projection scale is `1 / sqrt(128 * 64)`. Index Q and K use partial RoPE followed by a Hadamard rotation.

DeepSeek describes MXFP4 Q/K and BF16 scores for optimized production inference. The local GPU has no native floating FP4 arithmetic. This branch therefore uses an FP16 WMMA approximation only to identify a candidate set, then recomputes candidate scores with the original F32 query and F32 accumulation. The candidate margin was increased from 32 to 128 after paired quality measurements showed small real selection differences at margin 32.

## Strix Halo hardware

The target is a Radeon 8060S, RDNA 3.5, `gfx1151`:

| Property | Value |
| --- | ---: |
| Compute units | 40 |
| WGPs | 20 |
| SIMD32 units | 80 |
| Wave size | 32 |
| Maximum clock | 2.9 GHz |
| Physical LDS per WGP | 128 KiB |
| Maximum LDS per workgroup | 64 KiB |
| Physical VGPRs per SIMD | 1,536 |
| Wave slots per SIMD | 16 |
| L2 | 2 MiB |
| Infinity Cache | 32 MiB |
| Peak LPDDR5X bandwidth | 256 GB/s |

`gfx1151` supports GFX11 FP16/BF16 WMMA but not CDNA MFMA, FP8 matrix arithmetic, or floating FP4 matrix arithmetic. HIP uses wave32.

For this GPU, useful occupancy changes happen at resource thresholds rather than after small register reductions. The D512 sparse tile uses 40 KiB LDS, so 256 threads is better than 512 threads. The Indexer WMMA tile was changed from four heads to one head per tile, reducing LDS from about 52.7 KiB to about 39.2 KiB and increasing WGP residency from two to three workgroups.

## Implemented path

### Sorted Top-K

`ggml_top_k_sorted()` carries an explicit descending-value flag. CPU preserves descending order. HIP performs radix Top-K followed by a bitonic sort of the selected 512 IDs by score.

The score-order sort costs about 8.7 ms over all 21 CSA layers in the measured graph and is not a meaningful bottleneck.

### Split sparse IDs

The graph creates an I32 local-index input of shape:

```text
[128, queries per stream, 1, streams]
```

Local entries are physical raw-cache cells sorted by logical token position. Missing early-context cells are `-1`. They are concatenated with the 512 relative CSA IDs:

```text
[128 absolute raw IDs][512 relative CSA IDs]
```

`ggml_flash_attn_ext_add_sparse_indices_kv_split()` records the 128-entry split and the two-cache contract. The Flash Attention K argument is the raw K=V region and the V argument is the CSA K=V region. The HIP kernel maps the first ID segment into raw storage and the second segment relative to CSA storage. Shape checks ensure that both segments fit their physical cache ranges. The CPU reference implements the same segmented traversal.

The authoritative causal masks remain active. The redundant dense Top-K scatter mask was removed because sparse IDs already enforce membership; the original CSA visibility mask still rejects incomplete or future compressed rows.

### MQA head-major tile

The original sparse D512 tile grouped:

```text
4 query tokens x 8 query heads
```

The optimized sparse tile groups:

```text
1 query token x 32 query heads
```

All eight waves in a workgroup now use the same 640 IDs and the same shared MQA K/V rows. This substantially improves cache reuse without changing the selected set, QK reduction, softmax order, sinks, or output layout.

The final tile uses 256 threads, five 128-entry sparse iterations, and FP32 `P*V` accumulation. FP32 output accumulation follows the official TileLang reference and had no measurable throughput cost after the head-major retile.

### Cache precision

The deployed server uses F16 cache storage:

```text
-ctk f16 -ctv f16
```

This avoids full-cache Q8_0 dequantization in the current sparse launcher and retains more precision. DeepSeek V4 stores a shared K/V row, so the apparent K/V options refer to the same logical cache data.

## Performance

Benchmark command:

```bash
llama-bench \
  -m DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf \
  -p 2048 -n 0 -b 4096 -ub 2048 -ngl 99 -fa on -lm none
```

Latest controlled three-run comparison:

| Depth | Segmented raw/CSA | Concatenated raw+CSA | Change |
| ---: | ---: | ---: | ---: |
| 0 | 311.76 +/- 5.10 t/s | 308.66 +/- 3.10 t/s | +1.0% |
| 25,000 | 277.10 +/- 4.46 t/s | 271.63 +/- 3.05 t/s | +2.0% |
| 50,000 | 255.99 +/- 2.36 t/s | 251.89 +/- 2.52 t/s | +1.6% |

Earlier checkpoints measured 337.20 +/- 1.96 t/s at depth 0, 316.49 +/- 10.66 t/s at depth 10K, and 305.69 +/- 4.54 t/s at depth 20K. Large-model launch variance and cache state make the paired segmented comparison above the stronger evidence for the cache-layout change.

Segmented decode results:

| Depth | TG32 |
| ---: | ---: |
| 0 | 15.73 +/- 0.43 t/s |
| 25,000 | 14.60 +/- 0.72 t/s |
| 50,000 | 13.94 +/- 0.64 t/s |

The head-major kernel reduced measured sparse D512 attention from about 49.88 ms to 20.95 ms per CSA layer, a reduction of about 58%. It now costs approximately the same as the HCA D512 dense tile.

At depth 50K, the segmented profile measured about 26.71 ms per CSA sparse-attention dispatch, 22.85 ms per HCA attention dispatch, 17.07 ms for the approximate WMMA Indexer stage, and 13.79 ms for exact candidate reranking. Quantized MoE matrix multiplication remains the largest total cost.

The remaining final Top-K, score ordering, and `-INFINITY` fill account for less than 1% of a depth-50K pass. A compact 640-pair bitonic replacement was neutral at 50K and slower at 25K, so it was removed.

## Quality observations

Stock dense-mask output is not the architecture target, but it remains a useful numerical reference. On one paired 4,096-token sample with candidate margin 128 and FP32 sparse accumulation:

| Metric | Native sparse vs stock |
| --- | ---: |
| PPL ratio | 1.012828 +/- 0.007254 |
| Mean KLD | 0.034729 +/- 0.004352 |
| RMS probability delta | 4.129% +/- 0.239% |
| Same top token | 93.894% +/- 0.529% |

The FP16 output-accumulator ablation produced mixed results on the same sample: a lower PPL ratio and higher top-token agreement, but worse mean KLD. FP32 was retained because it follows the official sparse kernel, reduces aggregate distribution error, and has no measurable speed cost.

An eight-chunk FP16 ablation on `tests/test-backend-ops.cpp` produced PPL 1.975355 versus stock 1.977389, mean KLD 0.025220, RMS probability delta 4.748%, and 96.367% same-top agreement. The corresponding eight-chunk FP32 run was intentionally not completed after quality testing was stopped.

These comparisons do not provide official-reference logits. They measure finite-precision and accumulation-order differences against llama.cpp's stock dense-mask implementation.

## Validation

Deployed ROCm targeted tests:

```text
Segmented sparse FA: 2/2 passed
Sorted Top-K:       2/2 passed
Lightning Indexer general suite: 144/144 passed before deployment rebuild
```

The split tests cover:

```text
128 raw absolute IDs
512 CSA relative IDs
-1 raw sentinels
F16 and Q8_0 inputs
64:1 MQA head grouping
multiple streams
attention sinks
```

The llama-swap deployment passed its health check and an OpenAI-compatible chat-completion request.

## Rejected experiments

| Experiment | Result | Decision |
| --- | ---: | --- |
| D512 16 columns | 266.89 t/s at depth 20K | Rejected |
| D512 512 threads | 248.37 t/s at depth 20K | Rejected |
| Preconvert Indexer Q to F16 | 307.85 t/s | Rejected |
| Indexer two-head tile, stride 67 | 288.74 t/s | Rejected |
| Exact rerank 16 candidates/wave | 303.00 +/- 7.46 t/s | Rejected |
| Indexer candidate margin 256 | 307.63 t/s | Rejected; retained 128 |
| Compact 640-pair final Top-K | 273.84 t/s at 25K, 256.67 t/s at 50K | Rejected; no consistent gain |
| Full WMMA attention prototype | Large historical perplexity regression | Not used |

## Deployment

Persistent source and build paths inside the ROCm container:

```text
/opt/dsv4-origin-src
/opt/dsv4-origin-build
```

llama-swap config:

```text
/home/gaetan/models/llama-swap.yaml
```

The `deepseek-v4-flash:origin` entry runs:

```text
/opt/dsv4-origin-build/bin/llama-server
```

with a 65,536-token server context, batch 4,096, ubatch 2,048, one parallel slot, F16 cache, Flash Attention, and all layers on ROCm. llama-swap launches it in the persistent `rocm` container on an internal dynamic port and exposes it through port 8080. The llama-swap service is currently stopped by user request.

## Remaining work

1. Pass raw and CSA visibility masks separately if profiling shows their concat remains material; K/V cache segmentation is complete.
2. Add selected-row Q8_0 decode only if a lower-memory cache mode is required; F16 is currently better for both quality and this launcher's traffic.
3. Complete ordered Top-K behavior on non-HIP backends before treating `ggml_top_k_sorted()` as a portable public operation.
4. Add a direct compact-ID Lightning Indexer output only if a design also removes fill, scatter, and final full-range ranking; the isolated compact Top-K was not useful.
5. Run the complete ROCm backend suite before committing or proposing the change upstream.

## Primary references

- DeepSeek V4 paper: <https://arxiv.org/abs/2606.19348>
- DeepSeek V4 Flash 0731 config: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json>
- DeepSeek reference model: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/model.py>
- DeepSeek TileLang kernels: <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/kernel.py>
- DeepGEMM: <https://github.com/deepseek-ai/DeepGEMM>
- AMD GPU architecture specifications: <https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html>
- GPUOpen RDNA performance guide: <https://gpuopen.com/learn/rdna-performance-guide/>
