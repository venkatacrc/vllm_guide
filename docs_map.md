# vLLM Documentation Map (Topical Catalog)

This file catalogs every markdown document under the repo `docs/` tree, plus a few
important READMEs elsewhere, organized by **topic** instead of by folder. Use it
as a lookup: "which existing docs cover X?" Each entry gives file path + a
one-line description.

**Coverage note.** Some vLLM topics are covered well by existing docs (marked
✅). Others are only discoverable by reading source code (marked ⚠️); for those,
the day-files under `STUDY_GUIDE/dayNN.md` are the primary reading material.

---

## 1. Getting started & tour of the project

- ✅ `docs/README.md` — top-level docs index (used to build the site).
- ✅ `docs/getting_started/quickstart.md` — install → offline `LLM.generate` → OpenAI server.
- ✅ `docs/getting_started/installation/README.md` — install index; pick platform.
- ✅ `docs/getting_started/installation/gpu.md` + `gpu.cuda.inc.md`, `gpu.rocm.inc.md`, `gpu.apple.inc.md`, `gpu.xpu.inc.md` — device-specific install.
- ✅ `docs/getting_started/installation/cpu.md` + `cpu.x86.inc.md`, `cpu.arm.inc.md`, `cpu.apple.inc.md`, `cpu.s390x.inc.md` — CPU install.
- ✅ `docs/getting_started/installation/python_env_setup.inc.md` — Python venv setup snippet.

## 2. Architecture & internals

The core "how is vLLM built" material. Read these alongside the day files.

- ✅ `docs/design/arch_overview.md` — high-level architecture map. Complements Day 1.
- ✅ `docs/design/vllm_ir.md` — VLLM IR (unified dispatch/fusion/OOT extensibility).
- ✅ `docs/design/paged_attention.md` — historical explanation of PagedAttention (V0 kernel).
- ✅ `docs/design/prefix_caching.md` — canonical block-hash chain explainer.
- ✅ `docs/design/hybrid_kv_cache_manager.md` — design of the hybrid KV cache manager (sliding-window + full + mamba).
- ✅ `docs/design/attention_backends.md` — backend capability matrix (auto-generated).
- ✅ `docs/design/cuda_graphs.md`, `cuda_graphs_multimodal.md` — CUDA graph capture strategy.
- ✅ `docs/design/torch_compile.md`, `torch_compile_multimodal.md`, `debug_vllm_compile.md` — compile-mode integration.
- ✅ `docs/design/optimization_levels.md` — `-O0..-O3` and the passes each enables.
- ✅ `docs/design/fusions.md` — the pass table for fused custom ops.
- ✅ `docs/design/fused_moe_modular_kernel.md` — modular MoE kernel dispatch.
- ✅ `docs/design/moe_kernel_features.md` — MoE kernel compatibility matrix.
- ✅ `docs/design/model_runner_v2.md` — V2 model runner (rewrite; may or may not be live).
- ✅ `docs/design/multiprocessing.md` — spawn/fork/forkserver, why it matters.
- ✅ `docs/design/mm_processing.md` — `BaseMultiModalProcessor` and the MM pipeline.
- ✅ `docs/design/logits_processors.md` — logits-processor plugin design.
- ✅ `docs/design/custom_op.md` — writing a `vllm/torch.ops.vllm.*` custom op.
- ✅ `docs/design/dbo.md` — Dual Batch Overlap (MoE compute + all-to-all overlap).
- ✅ `docs/design/huggingface_integration.md` — how vLLM consumes HF configs/checkpoints.
- ✅ `docs/design/metrics.md` — Prometheus/OpenTelemetry emission model.
- ✅ `docs/design/nixl_kv_cache_lease.md` — NIXL P/D lease protocol.
- ✅ `docs/design/nixl_kv_push_connector.md` — push-based NIXL connector variant.
- ✅ `docs/design/io_processor_plugins.md` — IO-processor plugin API.
- ✅ `docs/design/lora_resolver_plugins.md` — LoRA-resolver plugin API.
- ✅ `docs/design/plugin_system.md` — the general plugin/entry-point system.

## 3. Model support & registration

- ✅ `docs/models/supported_models.md` — full model support matrix (which architectures work).
- ✅ `docs/models/generative_models.md` — decoder-only causal-LM usage.
- ✅ `docs/models/pooling_models/README.md` and siblings (`classify.md`, `embed.md`, `reward.md`, `scoring.md`, `specific_models.md`, `token_classify.md`, `token_embed.md`) — pooling / non-generative models.
- ✅ `docs/models/hardware_supported_models/cpu.md`, `xpu.md` — device-specific model gotchas.
- ✅ `docs/models/extensions/tensorizer.md`, `runai_model_streamer.md`, `fastsafetensor.md`, `instanttensor.md` — checkpoint streaming and loader extensions.
- ✅ `docs/contributing/model/README.md` — index of the "add-a-model" guides.
- ✅ `docs/contributing/model/basic.md` — dense/decoder model port template.
- ✅ `docs/contributing/model/multimodal.md` — multimodal model port template.
- ✅ `docs/contributing/model/registration.md` — how to register in `vllm/model_executor/models/registry.py`.
- ✅ `docs/contributing/model/tests.md` — required tests when adding a model.
- ✅ `docs/contributing/model/transcription.md` — ASR/transcription models.
- ⚠️ Model registry internals (dispatch, protocol enforcement) — read code (`vllm/model_executor/models/{registry,interfaces,interfaces_base}.py`) and **Day 2** of this guide.

## 4. Inference features (per-request behavior)

- ✅ `docs/features/README.md` — feature index page (starting point).
- ✅ `docs/features/multimodal_inputs.md` — how to pass images/audio/video in requests.
- ✅ `docs/features/prompt_embeds.md` — direct embedding inputs.
- ✅ `docs/features/tool_calling.md` — OpenAI tool-calling protocol support.
- ✅ `docs/features/structured_outputs.md` — grammar/JSON/regex-constrained generation.
- ✅ `docs/features/reasoning_outputs.md` — reasoning-model output parsing.
- ✅ `docs/features/interleaved_thinking.md` — interleaved reasoning outputs.
- ✅ `docs/features/lora.md` — LoRA adapter serving.
- ✅ `docs/features/custom_arguments.md` — custom `SamplingParams` extensions.
- ✅ `docs/features/custom_logitsprocs.md` — user-supplied logits processors.
- ✅ `docs/features/automatic_prefix_caching.md` — user-facing prefix cache guide.
- ✅ `docs/features/batch_invariance.md` — batch-invariance mode (`VLLM_BATCH_INVARIANT`).
- ✅ `docs/features/context_extension.md` — RoPE scaling / YaRN etc.
- ✅ `docs/features/sleep_mode.md` — sleep/wake to free GPU memory.
- ✅ `docs/features/index_cache.md` — index caching for RAG.

## 5. Speculative decoding

- ✅ `docs/features/speculative_decoding/README.md` — method selection guide + qualitative comparison.
- ✅ `docs/features/speculative_decoding/draft_model.md` — draft-model method.
- ✅ `docs/features/speculative_decoding/n_gram.md` — n-gram method.
- ✅ `docs/features/speculative_decoding/eagle.md` — EAGLE / EAGLE-3.
- ✅ `docs/features/speculative_decoding/mtp.md` — Multi-Token Prediction (DeepSeek).
- ✅ `docs/features/speculative_decoding/mlp.md` — MLP-based drafter.
- ✅ `docs/features/speculative_decoding/suffix.md` — suffix decoding.
- ✅ `docs/features/speculative_decoding/parallel_draft_model.md` — parallel draft variant.
- ✅ `docs/features/speculative_decoding/dynamic_speculative_decoding.md` — dynamic K.
- ✅ `docs/features/speculative_decoding/extract_hidden_states.md` — hidden-state export (for training EAGLE etc.).
- ✅ `docs/features/speculative_decoding/speculators.md` — `vllm-project/speculators` loader integration.
- ⚠️ Rejection sampling internals — read `vllm/v1/sample/rejection_sampler.py` and **Day 6** of this guide.
- ⚠️ `SpecDecodeMetadata` internals — read `vllm/v1/spec_decode/metadata.py` and **Day 6**.

## 6. Quantization

- ✅ `docs/features/quantization/README.md` — quantization index; which methods vLLM supports.
- ✅ `docs/features/quantization/auto_awq.md` — AWQ.
- ✅ `docs/features/quantization/bnb.md` — bitsandbytes.
- ✅ `docs/features/quantization/fp8_vit_attn.md` — FP8 vision-transformer attention.
- ✅ `docs/features/quantization/gguf.md` — GGUF (llama.cpp-style) loading.
- ✅ `docs/features/quantization/gptqmodel.md` — GPTQModel.
- ✅ `docs/features/quantization/inc.md` — Intel Neural Compressor.
- ✅ `docs/features/quantization/modelopt.md` — NVIDIA ModelOpt (`FP4`/`FP8` etc.).
- ✅ `docs/features/quantization/online.md` — online quantization at load time.
- ✅ `docs/features/quantization/quantized_kvcache.md` — quantized KV cache.
- ✅ `docs/features/quantization/quark.md` — AMD Quark.
- ✅ `docs/features/quantization/torchao.md` — torchao integration.
- ⚠️ `compressed-tensors` internals — read `vllm/model_executor/layers/quantization/compressed_tensors/` and **Day 8**.
- ⚠️ Per-layer method dispatch (`QuantizationConfig.get_quant_method`) — read `vllm/model_executor/layers/quantization/base_config.py` and **Day 8**.

## 7. Parallelism, distributed serving, KV connectors

- ✅ `docs/serving/parallelism_scaling.md` — TP/PP/DP overview.
- ✅ `docs/serving/data_parallel_deployment.md` — DP topology and DP-attention.
- ✅ `docs/serving/context_parallel_deployment.md` — context parallelism (CP).
- ✅ `docs/serving/expert_parallel_deployment.md` — expert parallelism (EP) + EPLB.
- ✅ `docs/serving/distributed_troubleshooting.md` — hangs, timeouts, NCCL debugging.
- ✅ `docs/features/disagg_prefill.md` — prefill-decode disaggregation.
- ✅ `docs/features/disagg_encoder.md` — encoder disaggregation for MM.
- ✅ `docs/features/kv_offloading_usage.md` — CPU/NVMe KV offload.
- ✅ `docs/features/nixl_connector_usage.md`, `nixl_connector_compatibility.md` — NIXL KV connector.
- ✅ `docs/features/mooncake_connector_usage.md`, `mooncake_store_connector_usage.md` — Mooncake connectors.
- ✅ `docs/features/moriio_connector_usage.md` — MoRIIO connector.
- ⚠️ `parallel_state` internals (TP/PP/DP/CP/EP groups) — read `vllm/distributed/parallel_state.py` and **Day 8**.
- ⚠️ EPLB scheduling — read `vllm/distributed/eplb/eplb_state.py` and **Day 8**.

## 8. Serving & deployment

- ✅ `docs/serving/offline_inference.md` — offline API (`LLM.generate`, `LLM.chat`).
- ✅ `docs/serving/online_serving/README.md` — OpenAI-compatible server index.
- ✅ `docs/serving/online_serving/openai_compatible_server.md` — full server API.
- ✅ `docs/serving/online_serving/generative_scoring.md` — scoring-with-generation trick.
- ✅ `docs/serving/online_serving/renderer.md` — chat/completion template renderer.
- ✅ `docs/serving/online_serving/speech_to_text.md` — Whisper-style transcription endpoints.
- ✅ `docs/serving/integrations/{langchain,llamaindex,claude_code,codex}.md` — client integrations.
- ✅ `docs/deployment/docker.md`, `k8s.md`, `nginx.md` — container / k8s / reverse-proxy.
- ✅ `docs/deployment/frameworks/*.md` (many files) — hosted-framework deployments (Anyscale, BentoML, RunPod, Modal, dstack, SkyPilot, Helm, LWS, Triton, HF Inference Endpoints, RAG frontends, dify, chatbox, streamlit, etc.).
- ✅ `docs/deployment/integrations/*.md` — production K8s stacks (AIBrix, Dynamo, kserve, kubeai, KubeRay, kaito, KThena, llamastack, llm-d, llmaz, production-stack).
- ✅ `docs/cli/README.md`, `docs/cli/serve.md`, `docs/cli/chat.md`, `docs/cli/complete.md`, `docs/cli/run-batch.md`, `docs/cli/launch/render.md`, `docs/cli/bench/{latency,mm_processor,serve,throughput}.md`, `docs/cli/json_tip.inc.md` — `vllm` CLI reference.

## 9. Configuration, tuning, environment

- ✅ `docs/configuration/README.md` — configuration overview.
- ✅ `docs/configuration/engine_args.md` — every `EngineArgs` flag.
- ✅ `docs/configuration/serve_args.md` — server-only flags.
- ✅ `docs/configuration/env_vars.md` — all `VLLM_*` env vars.
- ✅ `docs/configuration/optimization.md` — performance-tuning guide (batch, cache, compile).
- ✅ `docs/configuration/model_resolution.md` — HF Hub → local resolution.
- ✅ `docs/configuration/conserving_memory.md` — reducing KV / weight memory.
- ✅ `docs/usage/README.md` — usage index (older content).
- ✅ `docs/usage/v1_guide.md` — **V1 vs V0** migration guide; canonical entry to V1.
- ✅ `docs/usage/faq.md`, `troubleshooting.md`, `metrics.md`, `usage_stats.md`, `reproducibility.md`, `security.md` — operational docs.

## 10. Training-adjacent tooling in this repo

vLLM is an inference engine; training code lives elsewhere. But these docs
describe how vLLM plugs into RLHF/PPO/GRPO frameworks that need fast rollouts:

- ✅ `docs/training/async_rl.md` — asynchronous RL loops.
- ✅ `docs/training/rlhf.md` — RLHF integration examples.
- ✅ `docs/training/trl.md` — HuggingFace TRL integration.
- ✅ `docs/training/layerwise.md` — layerwise checkpoint loading during training.
- ✅ `docs/training/weight_transfer/README.md`, `base.md`, `ipc.md`, `nccl.md` — how to hot-swap weights from trainer → inference engine.
- ⚠️ Draft-model training — **not in this repo**; see **Day 9** for external references (`vllm-project/speculators`, SafeAILab/EAGLE, FasterDecoding/Medusa, IBM Granite spec, AMD PARD, Arctic Inference suffix decoding).

## 11. Benchmarking

- ✅ `docs/benchmarking/README.md` — benchmarking overview.
- ✅ `docs/benchmarking/cli.md` — `vllm bench` CLI.
- ✅ `docs/benchmarking/dashboard.md` — result dashboards.
- ✅ `docs/benchmarking/sweeps.md` — parameter sweeps.

## 12. API reference & examples

- ✅ `docs/api/README.md` — auto-generated Python API index.
- ✅ `docs/examples/README.md` — index into `examples/` (offline, online, LoRA, spec, MM, etc.).

## 13. Contributing / CI / dev process

- ✅ `docs/contributing/README.md` — new-contributor tour.
- ✅ `docs/contributing/ci/failures.md`, `nightly_builds.md`, `update_pytorch_version.md` — CI hygiene.
- ✅ `docs/contributing/dockerfile/dockerfile.md` — Dockerfile layout.
- ✅ `docs/contributing/incremental_build.md` — faster iterative builds.
- ✅ `docs/contributing/profiling.md` — how to profile.
- ✅ `docs/contributing/deprecation_policy.md` — API deprecation lifecycle.
- ✅ `docs/contributing/editing-agent-instructions.md` — how to change AGENTS.md / guides.
- ✅ `docs/contributing/vulnerability_management.md` — security disclosure process.
- ✅ `docs/governance/{collaboration,committers,process}.md` — project governance.
- ✅ `docs/community/{contact_us,meetups,sponsors}.md` — community.

---

## Where existing docs are **thin** (rely on Day files)

| Topic | Primary source | Day file |
| --- | --- | --- |
| Request lifecycle end-to-end | Code | Day 1 & Day 10 |
| Model registry lookup + protocol enforcement | Code | Day 2 |
| Attention custom op / backend selector | Code | Day 4 |
| KV cache manager & block pool internals | Code + `hybrid_kv_cache_manager.md` | Day 4 |
| Scheduler control flow (`schedule()` step body) | Code | Day 5 |
| Speculative decoding proposer + verifier plumbing | Code + `speculative_decoding/*.md` | Day 6 & 7 |
| Rejection sampler math | Code | Day 6 |
| Per-method spec-decoding classes | Code + `speculative_decoding/*.md` | Day 7 |
| Per-layer quantization dispatch (`get_quant_method`) | Code | Day 8 |
| Parallel groups (`parallel_state.py`) | Code + `parallelism_scaling.md` | Day 8 |
| Structured output backend selection | Code + `structured_outputs.md` | Day 9 |
| Feature interaction matrix | Scattered | Day 10 |

## Suggested doc reading order (systems eng, from scratch)

1. `docs/getting_started/quickstart.md`
2. `docs/usage/v1_guide.md`
3. `docs/design/arch_overview.md`
4. `docs/design/paged_attention.md`
5. `docs/design/prefix_caching.md`
6. `docs/design/hybrid_kv_cache_manager.md`
7. `docs/design/attention_backends.md`
8. `docs/design/torch_compile.md` + `optimization_levels.md`
9. `docs/design/multiprocessing.md`
10. `docs/design/mm_processing.md`
11. `docs/features/speculative_decoding/README.md`
12. `docs/features/structured_outputs.md`
13. `docs/features/quantization/README.md`
14. `docs/serving/parallelism_scaling.md`
15. `docs/design/vllm_ir.md`

That's ~15 docs, most short — good complement to the 10-day study guide.
