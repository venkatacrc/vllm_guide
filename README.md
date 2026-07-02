# vLLM Offline Study Guide (10 days, ~1 hour/day)

A self-contained study guide to the vLLM codebase, written to disk so that a
reader with no internet access can learn the whole system by opening only
files in this repo. Nothing here depends on external links; every reference is
either to a file inside `/Users/v0c03vo/code/vllm` (with line numbers) or to
another file in `STUDY_GUIDE/`.

Written from source-code exploration and the existing `docs/` tree as of the
tip of the current working checkout. Where the codebase was ambiguous, the
day files flag it as **"open question"** rather than guessing.

---

## How to use this guide

- **One day at a time.** Each `dayNN.md` is designed for a **single 45–60 minute
  session**. Read start to finish, then do the exercise at the bottom.
- **Have your editor open** in `/Users/v0c03vo/code/vllm` while you read. Every
  reference is `path/to/file.py:LINE_NUMBER`. Jump to those lines and confirm
  the day file's claims with your own eyes — this is the highest-leverage
  study technique.
- **Read in order.** Later days assume earlier days. Day 10 in particular is
  a synthesis that cross-links to all previous days.
- **Skip the exercises at your peril.** They're not filler; they force you to
  actually navigate the code rather than passively reading prose.
- **Cross-reference existing docs.** `docs_map.md` shows which official docs
  cover which topics. Where the docs already cover something well, the day
  files summarize rather than re-explain.

---

## Curriculum at a glance

| Day | Topic | Primary code touched |
| --- | --- | --- |
| [Day 1](day01.md) | The big picture: request lifecycle & V1 architecture | `vllm/entrypoints/*`, `vllm/v1/engine/*`, `vllm/v1/executor/*`, `vllm/v1/worker/*` |
| [Day 2](day02.md) | Model registry + dense transformer walkthrough (Llama) | `vllm/model_executor/models/{registry,interfaces,interfaces_base,llama}.py`, `vllm/model_executor/layers/{linear,layernorm,vocab_parallel_embedding,rotary_embedding}` |
| [Day 3](day03.md) | MoE + multimodal + hybrid architectures (Mixtral, LLaVA, Mamba/Jamba) | `vllm/model_executor/models/{mixtral,llava,mamba,jamba}.py`, `vllm/model_executor/layers/fused_moe/`, `vllm/multimodal/*` |
| [Day 4](day04.md) | KV cache management & PagedAttention | `vllm/v1/core/{kv_cache_manager,kv_cache_coordinator,single_type_kv_cache_manager,block_pool,kv_cache_utils}.py`, `vllm/model_executor/layers/attention/*`, `vllm/v1/attention/*` |
| [Day 5](day05.md) | Scheduling: continuous batching, chunked prefill, prefix caching, preemption | `vllm/v1/core/sched/*`, `vllm/v1/request.py` |
| [Day 6](day06.md) | Speculative decoding: framework and control flow | `vllm/v1/spec_decode/{metadata,llm_base_proposer}.py`, `vllm/v1/sample/rejection_sampler.py`, `vllm/v1/engine/core.py` |
| [Day 7](day07.md) | Speculative decoding methods compared (n-gram / suffix / draft / EAGLE / Medusa / MTP / DFlash / dynamic) | `vllm/v1/spec_decode/*.py`, `vllm/config/speculative.py`, `vllm/model_executor/models/*eagle*.py`, `*mtp*.py`, `medusa.py` |
| [Day 8](day08.md) | Quantization + parallelism (TP / PP / DP / EP / CP) | `vllm/model_executor/layers/quantization/*`, `vllm/distributed/*`, `vllm/model_executor/layers/linear.py` |
| [Day 9](day09.md) | Draft-model training (external) + structured/guided decoding | `vllm/transformers_utils/configs/speculators/*`, `vllm/v1/structured_output/*` |
| [Day 10](day10.md) | Synthesis: full end-to-end trace + open questions | Everything |

---

## Also in this directory

- [`docs_map.md`](docs_map.md) — topical catalog of the entire `docs/` tree, plus
  a note on which topics are underdocumented (and therefore relied on the day
  files instead).

---

## What this guide is not

- **Not a replacement for reading source.** It's a *guided tour with a map*.
  You still have to open the files.
- **Not a training/RL guide.** vLLM is an inference engine. See Day 9 for what
  the repo does and does not cover on training draft models.
- **Not comprehensive.** There are ~1M+ LOC in this repo. The guide covers the
  concepts you *have* to understand to be effective, plus enough breadth to
  navigate the rest.
- **Not a benchmarking guide.** See `docs/benchmarking/` for that.

---

## Glossary

vLLM-specific terms and abbreviations used throughout the guide.

### Core objects

- **`LLM`** — user-facing offline API (`vllm/entrypoints/llm.py`). Wraps `LLMEngine`.
- **`LLMEngine`** — synchronous V1 engine (`vllm/v1/engine/llm_engine.py`). Single-process.
- **`AsyncLLM`** — asynchronous V1 engine (`vllm/v1/engine/async_llm.py`). Split across frontend + `EngineCore` processes.
- **`EngineCore`** — core stepping loop (`vllm/v1/engine/core.py`). Owns the scheduler + executor; usually runs in its own process.
- **`Executor`** — abstraction for running the model on N workers. `UniprocExecutor` (in-process), `MultiprocExecutor` (spawn workers), `RayDistributedExecutor` (Ray). See `vllm/v1/executor/`.
- **`Worker`** — per-GPU process. Owns the model, KV cache, model runner. `vllm/v1/worker/gpu_worker.py`.
- **`GPUModelRunner`** — the *big* class (~4700 lines). Prepares inputs, runs forward, samples, handles spec-decode drafting. `vllm/v1/worker/gpu_model_runner.py`.
- **`Scheduler`** — decides which requests run each step and how many tokens each gets. `vllm/v1/core/sched/scheduler.py`.
- **`Request`** — per-request state machine in the engine. `vllm/v1/request.py`.
- **`InputProcessor` / `OutputProcessor`** — front-end tokenization/detokenization and prompt/output shaping. `vllm/v1/engine/{input_processor,output_processor}.py`.

### Configuration

- **`VllmConfig`** — the root aggregate config passed everywhere. `vllm/config/vllm.py`.
- **`ModelConfig`, `CacheConfig`, `ParallelConfig`, `SchedulerConfig`, `SpeculativeConfig`, `StructuredOutputsConfig`, …** — subconfigs, one per subsystem.
- **`EngineArgs`** — CLI/entry-point-level flags that get resolved into `VllmConfig`. `vllm/engine/arg_utils.py`.

### KV cache & attention

- **PagedAttention** — the KV-cache-in-fixed-size-blocks-with-a-block-table paradigm. In V1, the *technique*; the actual kernel is provided by the attention backend (FlashAttention, Triton, FlashInfer, or MLA).
- **Block** — a contiguous chunk of KV cache holding tokens for one logical KV group at fixed size (`block_size`, default 16 tokens).
- **Block table** — per-request array of physical block ids: index `i` gives the block for logical position `i*block_size`.
- **Slot mapping** — flat array mapping each *token position in this step* to a physical *slot* (`block_id * block_size + offset_within_block`). Used by the write path.
- **`BlockPool`** — the free-block allocator (LRU-ish; `vllm/v1/core/block_pool.py`).
- **`KVCacheManager`** — high-level KV cache API (`vllm/v1/core/kv_cache_manager.py`).
- **`KVCacheCoordinator`** — multi-group coordinator (`UnitaryKVCacheCoordinator`, `HybridKVCacheCoordinator`, `KVCacheCoordinatorNoPrefixCache`).
- **`SingleTypeKVCacheManager`** — per-cache-type manager (`FullAttentionManager`, `SlidingWindowManager`, `MambaManager`, `ChunkedLocalAttentionManager`, `CrossAttentionManager`).
- **`KVCacheSpec`** — declarative "what does this layer need in cache". `FullAttentionSpec`, `SlidingWindowSpec`, `MambaSpec`, `ChunkedLocalAttentionSpec`, etc. `vllm/v1/kv_cache_interface.py`.
- **`unified_attention_with_output`** — the vLLM custom op every attention layer calls. See `vllm/model_executor/layers/attention/attention.py`.
- **KV connector** — pluggable inter-node KV transfer (NIXL, Mooncake, MoRIIO). Enables prefill-decode disaggregation. `vllm/distributed/kv_transfer/*`.

### Attention backends

- **FA / FlashAttention** — `vllm/v1/attention/backends/flash_attn.py`. Default on Hopper/H100/H200.
- **Triton** — Triton implementation (`triton_attn.py`), portable fallback.
- **FlashInfer** — third-party backend (`flashinfer.py`).
- **MLA / Multi-head Latent Attention** — DeepSeek's compressed-latent attention. Backends: `mla/{common,flashmla,triton_mla,flashattn_mla,flashinfer_mla}.py`.
- **Backend selector** — `vllm/v1/attention/backend/selector.py`, decides which backend to use per model + hardware.

### Batching & scheduling

- **Continuous batching** — new requests join a running batch every step; no fixed batch size. The default V1 behavior.
- **Chunked prefill** — prefills longer than the per-step token budget are split into multiple prefill "chunks". Enabled by default in V1.
- **Prefix caching** — reuse KV blocks across requests when prompt prefixes match, keyed by rolling block hash. `vllm/v1/core/kv_cache_utils.py` computes hashes; `BlockPool` maintains the cache.
- **Preemption** — when GPU memory runs out, a running request is bumped back to `WAITING`/`PREEMPTED` and its blocks are freed. See `Scheduler._preempt_lowest_priority_request`.
- **Priority scheduling** — supports `Request.priority` for FCFS-with-priority. `RequestQueue` variants: `FCFSRequestQueue`, `PriorityRequestQueue`.
- **Persistent batch** — the model runner's stable request→row mapping; `InputBatch` in `vllm/v1/worker/input_batch.py`.

### Sampling

- **`Sampler`** — final logits → token id sampler. `vllm/v1/sample/sampler.py`. Handles temperature, top-k, top-p, min-p, seeds.
- **`LogitsProcessor`** — pre-sampler transforms (bias, repetition penalty, guided-decoding bitmask). `vllm/model_executor/layers/logits_processor.py` and pluggable via `vllm/v1/sample/logits_processor/`.
- **`RejectionSampler`** — speculative decoding verifier. `vllm/v1/sample/rejection_sampler.py`.

### Speculative decoding

- **Proposer / drafter** — cheap fast source of candidate tokens. Base class `SpecDecodeBaseProposer` in `vllm/v1/spec_decode/llm_base_proposer.py`.
- **Verifier** — the target model + rejection sampler; verifies the drafter's proposals.
- **`SpecDecodeMetadata`** — the bookkeeping bundle passed alongside spec-decode forward passes. `vllm/v1/spec_decode/metadata.py`.
- **EAGLE / EAGLE-3** — draft-head architecture reusing target hidden states. `vllm/v1/spec_decode/eagle.py`, model modules under `vllm/model_executor/models/llama_eagle{,3}.py` etc.
- **Medusa** — multi-head draft on top of the frozen target. `vllm/model_executor/models/medusa.py`, drafter `vllm/v1/spec_decode/medusa.py`.
- **MTP (Multi-Token Prediction)** — DeepSeek's built-in speculative heads. `vllm/model_executor/models/deepseek_mtp.py` and `*_mtp.py` variants.
- **N-gram** — non-model prompt-lookup speculation. `vllm/v1/spec_decode/ngram_proposer.py` and GPU variant.
- **Suffix decoding** — token-suffix trie speculation. `vllm/v1/spec_decode/suffix_decoding.py`.
- **Draft-model** — full independent smaller model as drafter. `vllm/v1/spec_decode/draft_model.py`.
- **DFlash** — Data-parallel flash speculative decoding. `vllm/v1/spec_decode/dflash.py`.
- **Dynamic SD** — runtime-adaptive K. `vllm/v1/spec_decode/dynamic/*`.
- **Speculators** — the `vllm-project/speculators` format for draft checkpoints; loader configs in `vllm/transformers_utils/configs/speculators/`.

### Quantization

- **`QuantizationConfig`** — base class in `vllm/model_executor/layers/quantization/base_config.py`. Registered per method in `__init__.py`.
- **Quant methods**: `fp8`, `gptq`, `gptq_marlin`, `awq`, `awq_marlin`, `compressed-tensors`, `bitsandbytes`, `bnb`, `gguf`, `modelopt`, `modelopt_fp8`, `modelopt_fp4`, `torchao`, `quark`, `hqq`, `inc`, `deepspeedfp`, `experts_int8`, `neuron_quant`, `moe_wna16`, `ptpc_fp8`, `rtn`, `tpu_int8`.
- **`LinearMethodBase` / `Fp8LinearMethod` / …** — per-linear-layer method interface. `vllm/model_executor/layers/quantization/{fp8,...}.py`.
- **KV-cache quant** — `int8_kv_cache`, `fp8_kv_cache`, `fp8_e5m2_kv_cache`; see `docs/features/quantization/quantized_kvcache.md`.
- **compressed-tensors** — sub-registry for the `compressed-tensors` format (mixed methods within one checkpoint). `vllm/model_executor/layers/quantization/compressed_tensors/`.

### Parallelism

- **TP (tensor parallel)** — split each linear layer across ranks; row/column parallel. `RowParallelLinear`, `ColumnParallelLinear`, `MergedColumnParallelLinear`, `QKVParallelLinear`.
- **PP (pipeline parallel)** — split layers across ranks; send hidden states between stages. `SupportsPP` protocol.
- **DP (data parallel)** — replicate model across process groups, each handles a subset of requests. `parallel_state.py` DP groups.
- **EP (expert parallel)** — for MoE, split experts across ranks (with or without TP). `FusedMoE` respects EP.
- **CP (context parallel)** — split sequence dimension across ranks.
- **EPLB** — expert-parallel load balancing (`vllm/distributed/eplb/`). Detects hot experts and moves them.
- **`parallel_state.py`** — the central process-group ownership (`vllm/distributed/parallel_state.py`).
- **Device communicators** — abstraction over NCCL/HCCL/GLOO/PyNCCL/XCCL/etc. `vllm/distributed/device_communicators/`.

### Multimodal

- **`MULTIMODAL_REGISTRY`** — global registry of processors/handlers per model. `vllm/multimodal/registry.py`.
- **`BaseMultiModalProcessor`** — abstract processor mapping raw inputs (PIL images, audio, video) to token embeddings + placeholders. `vllm/multimodal/processing.py`.
- **`MultiModalKwargs`** — the container for image/audio/video tensors passed to `forward`. `vllm/multimodal/inputs.py`.
- **`embed_multimodal`** — helper for scattering visual tokens into the text sequence. Used inside model `forward` methods.
- **Encoder disagg** — separating the vision-encoder step from decode (`docs/features/disagg_encoder.md`).

### Structured / guided decoding

- **`StructuredOutputManager`** — top-level orchestrator. `vllm/v1/structured_output/__init__.py`.
- **`StructuredOutputBackend`** — abstract (`vllm/v1/structured_output/backend_types.py`).
- **Backends** — `backend_xgrammar.py`, `backend_outlines.py`, `backend_lm_format_enforcer.py`, `backend_guidance.py` (`llguidance`).
- **`StructuredOutputGrammar`** — per-request compiled FSM/grammar. Advances with each token, produces a per-token bitmask.
- **`apply_grammar_bitmask`** — where the bitmask is added to logits, inside the model runner (`gpu_model_runner.py`).
- **Reasoner** — extract reasoning-model outputs (`vllm/reasoning/*`); can defer structured constraints until reasoning tags close.

### Custom ops

- **`torch.ops.vllm.unified_attention_with_output`** — the read/write attention op. `attention.py`.
- **`torch.ops.vllm.unified_kv_cache_update`** — KV write.
- **`set_forward_context(...)`** — thread-local context giving custom ops access to per-layer metadata without changing `forward` signatures. `vllm/forward_context.py`.

### Miscellany

- **CUDA graph** — captured static graph of one decode step, replayed for speed. `vllm/v1/cudagraph_dispatcher.py`, `vllm/compilation/*`.
- **torch.compile / `-O2`** — Inductor compile pass; `docs/design/torch_compile.md`, `optimization_levels.md`.
- **vLLM IR** — internal representation of ops for the compile pipeline. `docs/design/vllm_ir.md`.
- **Fusion pass** — Inductor passes that fuse quant + norm + attention pieces. `vllm/compilation/`, `docs/design/fusions.md`.
- **`SchedulerOutput`** — the batch descriptor produced by `Scheduler.schedule()` and consumed by the executor.
- **`ModelRunnerOutput`** — the batch results returned by the model runner.
- **`EngineCoreRequest` / `EngineCoreOutputs`** — serializable messages crossing the ZMQ boundary between frontend and `EngineCore`.
- **`RequestOutputCollector`** — the queue draining tokens for a single request in `AsyncLLM`.

### V1 vs V0

- **V0** — the original engine (`vllm/engine/*`, `vllm/worker/*`), retained for a few paths but deprecated. Some docs still reference V0 pieces (`docs/design/paged_attention.md` explains the V0 CUDA kernel).
- **V1** — the current re-architecture (`vllm/v1/*`). Everything in this study guide is V1 unless noted.

---

## Prerequisites (what to already know)

- Modern transformer-decoder LM architecture (Q/K/V, softmax attention, KV cache, RoPE).
- PyTorch basics: `nn.Module`, `forward`, dispatch, autograd (autograd is off during inference).
- Python asyncio at a basic level.
- Rough understanding of NCCL collectives (allreduce, allgather, all-to-all).
- What a CUDA graph is at a conceptual level.

If any of these are shaky, brush up before starting Day 4; you can survive Days 1–3 without them.

---

## After the 10 days

Suggested continuations:

1. Trace a **single non-Llama architecture** you care about (Qwen? DeepSeek?
   Mistral?) end-to-end using the patterns from Days 2–3.
2. Modify **one small thing** — e.g. add a `SamplingParams` field and thread
   it through `InputProcessor → Scheduler → Sampler`. Or add a synthetic
   spec-decode method that just returns constant tokens. Making it work is
   the fastest way to feel confident.
3. Read `docs/design/vllm_ir.md`, then read `vllm/compilation/` end-to-end.
4. Read `docs/design/dbo.md`, then trace DBO through `FusedMoE`.
5. Play with `VLLM_LOGGING_LEVEL=DEBUG` and re-trace a request; the log lines
   correspond to exactly the code paths in the day files.

Good luck. Ten days of one hour each is enough to move from "I've heard of
vLLM" to "I can navigate any file in this repo and know where it fits." Take
notes; write questions in the margins; open the code every time the day file
gives you a `file.py:LINE` reference. That's how it sticks.
