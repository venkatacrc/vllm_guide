# Day 2 — Model Registry & Dense Transformer Walkthrough

**By the end of today you will understand:** how a HuggingFace `config.architectures` string turns into a concrete `nn.Module`, what the shared abstractions are that every model conforms to (attention layer, linear layers, weight loader, capability protocols), and — using LLaMA as a worked example — how a real vLLM model is wired together.

> Time budget: ~60 minutes.

## 1. From HF checkpoint to Python class

The lookup happens in `vllm/model_executor/model_loader/utils.py`:

```193:225:vllm/model_executor/model_loader/utils.py
def _get_model_architecture(model_config):
    architectures = getattr(model_config.hf_config, "architectures", None) or []
    model_cls, arch = model_config.registry.resolve_model_cls(
        architectures, model_config=model_config)
    ...
```

The `registry` here is the global `ModelRegistry` singleton exposed at `vllm/model_executor/models/registry.py:1396` (built by `_ModelRegistry()`).

### 1a. Where the mapping table lives

Everything is in one file — `vllm/model_executor/models/registry.py`. Skim these anchor points:

| Table | Line | Contents |
| --- | --- | --- |
| `_TEXT_GENERATION_MODELS` | 71–208 | ~140 decoder-only causal LMs, keyed by HF architecture string, value = `(module_relname, class_name)` |
| `_EMBEDDING_MODELS` | 210–262 | Embedding-only models |
| `_MULTIMODAL_MODELS` | 331–580 | VLMs, audio models, ASR |
| `_SPECULATIVE_DECODING_MODELS` | 582–633 | EAGLE / EAGLE-3 / MTP / Medusa heads |
| `_TRANSFORMERS_BACKEND_MODELS` | 635–680 | Fallback to pure HF Transformers |
| `_PREVIOUSLY_SUPPORTED_MODELS` | 701–736 | Emits "removed in vX.Y" errors |
| `_OOT_SUPPORTED_MODELS` | 738–743 | Out-of-tree plugin URLs |

Each entry ultimately becomes a `_LazyRegisteredModel` (line 832) so importing `vllm` doesn't have to import all 288 model files. A subprocess is used for inspection (`_LazyRegisteredModel.inspect_model_cls` at `:917`) to avoid triggering CUDA init.

### 1b. The lookup itself

`_ModelRegistry.resolve_model_cls(architectures, model_config)` at `vllm/model_executor/models/registry.py:1244` is the function you should be able to describe from memory:

1. `_normalize_arch(...)` at `:1166` — handles suffix rewrites (e.g. `ForCausalLM` → `ForSequenceClassification` when the user set `--task classify`).
2. Look up in `_VLLM_MODELS` (`:682`, union of every table above).
3. On miss, try `_try_resolve_transformers` (`:1096`) — falls back to the HF `transformers` backend, or `trust_remote_code=True` with `auto_map`.
4. On failure, `_raise_for_unsupported` (`:1051`) produces the "removed in vX.Y" or OOT-plugin error.

### 1c. Registering a new architecture

Public API:

```1005:1049:vllm/model_executor/models/registry.py
def register_model(self, model_arch, model_cls):
    """Register an external model to be used in vLLM."""
    ...
```

Two shapes:

- Direct class: `ModelRegistry.register_model("MyForCausalLM", MyForCausalLM)`.
- Lazy path: `ModelRegistry.register_model("MyForCausalLM", "my_pkg.my_module:MyForCausalLM")`.

Doc: `docs/contributing/model/registration.md`.

## 2. The common abstractions every model conforms to

### 2a. Base protocols (no keyword `abstract`)

`vllm/model_executor/models/interfaces_base.py`:

- `VllmModel(Protocol)` at line 46 — requires `__init__(vllm_config, prefix="")` (line 50), `embed_input_ids(input_ids)` (line 52), and `forward(input_ids, positions)` (line 56). This is the base contract for **every** vLLM model.
- `VllmModelForTextGeneration(VllmModel)` at line 113 — adds `compute_logits(hidden_states)` (line 122).
- `VllmModelForPooling(VllmModel)` at line 147 — for embedding / classification / reward / scoring models; class vars set the default pooling type and score type.

Predicates the registry consults: `is_text_generation_model` (line 135), `is_pooling_model` (line 223).

**Important:** the vLLM `forward` signature is **`(input_ids, positions, intermediate_tensors=None, inputs_embeds=None)`**. There is no per-forward `kv_caches` or `attn_metadata` argument in V1. The `Attention` module (which we get to in §4b) fetches those from the forward context.

### 2b. Capability mixins

`vllm/model_executor/models/interfaces.py` defines a menagerie of `runtime_checkable` protocols. A model class opts in by inheriting the protocol and setting the sentinel class variable (`supports_lora=True`, `supports_pp=True`, etc.). The registry inspects the class once (in the lazy-import subprocess) to populate a `_ModelInfo`. Learn these names:

| Protocol | Line | Meaning |
| --- | --- | --- |
| `SupportsMultiModal` | 93 | The model accepts image/video/audio; requires `embed_multimodal`, `get_placeholder_str`. |
| `SupportsMultiModalPruning` | 412 | Encoder can drop redundant image tokens before the LLM. |
| `SupportsLoRA` | 538 | Declares `packed_modules_mapping`, `embedding_modules`, `lora_manager`. |
| `SupportsPP` | 617 | `make_empty_intermediate_tensors` + `forward(..., intermediate_tensors=)`. |
| `HasInnerState` | 736 | Has non-KV state (e.g. Mamba conv/temporal buffers). |
| `IsAttentionFree` | 762 | Pure SSM, no attention KV cache. |
| `IsHybrid` | 789 | Mixes attention with SSM; requires `get_mamba_state_shape_from_config`. |
| `MixtureOfExperts` | 846 | Exposes `moe_layers`, `num_logical_experts`, EPLB hooks (`set_eplb_state`). |
| `SupportsMambaPrefixCaching` | 952 | Mamba state variant that supports prefix caching. |
| `SupportsQuant` | 999 | Concrete class (not Protocol); helps quant configs find `packed_modules_mapping` and mapper. |
| `SupportsTranscription` | 1077 | Whisper-style ASR / translation. |
| `SupportsEagle` / `SupportsEagle3` | 1343 / 1373 | Target model exposes hooks for EAGLE drafters (Day 7). |
| `SupportsMRoPE` | 1448 | 3D rotary positions for Qwen-VL-style models. |
| `SupportsEncoderCudaGraph` | 1547 | Vision tower is CUDA-graph capturable. |

Each protocol has a matching `supports_*` / `has_*` / `is_*` free function you can use to test a class at runtime.

## 3. The shared layer library

Before you can read a model file you need to know what building blocks it consumes. All under `vllm/model_executor/layers/`:

- `linear.py` — `LinearBase`, `ReplicatedLinear`, `ColumnParallelLinear`, `RowParallelLinear`, `MergedColumnParallelLinear`, `QKVParallelLinear`. All accept a `quant_config` in their `__init__`, which selects the `LinearMethodBase` (Day 8).
- `vocab_parallel_embedding.py` — `VocabParallelEmbedding` (line 198) and `ParallelLMHead` (line 505). Shards the vocab dim across TP.
- `layernorm.py` — `RMSNorm` (line 37) with a fused (`hidden_states, residual`) two-return-value calling convention that models rely on.
- `rotary_embedding/` — `get_rope(head_dim, max_position, rope_parameters, is_neox_style)` factory.
- `activation.py` — `SiluAndMul`, `get_act_fn`.
- `attention/attention.py` — `Attention` (line 192), the standard MHA/MQA/GQA module.
- `attention/mla_attention.py` — `MLAAttention` (line 339), the DeepSeek-V2/V3/V4 latent-attention variant.
- `attention/encoder_only_attention.py` — `EncoderOnlyAttention` (line 51), no-KV variant for encoder passes.
- `fused_moe/layer.py` — `FusedMoE` factory (line 100) plus expert kernel dispatch. Day 3.

## 4. Walkthrough: `vllm/model_executor/models/llama.py`

This is the reference dense-transformer implementation. Anyone porting a new decoder-only model uses this as the template.

### 4a. LlamaMLP

```79:119:vllm/model_executor/models/llama.py
class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act, quant_config, ...):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(...)      # line 92
        self.down_proj = RowParallelLinear(...)                  # line 100
        ...
        self.act_fn = SiluAndMul()                               # line 113
```

`MergedColumnParallelLinear` fuses `gate_proj` and `up_proj` into a single column-parallel matmul (the two-column concat is later split by `SiluAndMul`).

### 4b. LlamaAttention

`vllm/model_executor/models/llama.py:122`:

- **QKV**: `QKVParallelLinear(hidden_size, head_dim, total_num_heads, total_num_kv_heads, ...)` at line 162 — one fused kernel producing (Q, K, V) shards for this TP rank.
- **Output**: `RowParallelLinear(...)` at line 172.
- **Rotary**: `_init_rotary_emb` at line 233 calls `get_rope(head_dim, max_position, rope_parameters=..., is_neox_style=True)`.
- **Attention module**: `attn_cls = EncoderOnlyAttention if attn_type == ENCODER_ONLY else Attention` at line 203, then

```209:219:vllm/model_executor/models/llama.py
        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
        )
```

- **Forward** (line 221):

```221:231:vllm/model_executor/models/llama.py
    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([...], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output
```

Note there is no `kv_cache` argument. Inside `Attention.forward` (`vllm/model_executor/layers/attention/attention.py:452`), the KV cache and attention metadata are pulled from the forward context via `get_attention_context(layer_name)`. Day 4 dives into that mechanism.

### 4c. LlamaDecoderLayer

`llama.py:248` — the standard "input_layernorm → attn → post_attention_layernorm → mlp" with a fused-residual RMSNorm calling convention:

```310:327:vllm/model_executor/models/llama.py
    def forward(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
```

The `(hidden_states, residual)` two-return-value form is used by every vLLM decoder layer so that the fused RMSNorm+residual+quantization compile pass can find it (see `docs/design/fusions.md`).

### 4d. LlamaModel — the transformer stack

`llama.py:334`:

- `@support_torch_compile(...)` decorator at line 334 marks the class as a torch.compile boundary.
- Inherits `EagleModelMixin` (line 344).
- **Weight name mapper** at line 345:

```345:354:vllm/model_executor/models/llama.py
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        },
    )
```

This is the exact HF-to-vLLM weight-name rewrite: three separate `q_proj`/`k_proj`/`v_proj` HF tensors are stacked into one `qkv_proj`; `gate_proj` and `up_proj` into `gate_up_proj`.

- `__init__` at `:356` — PP-aware:
  - Only PP rank 0 creates the `embed_tokens` (line 373).
  - Only PP last rank creates the `norm` and (for `LlamaForCausalLM`) `lm_head` (line 388).
  - `make_layers(config.num_hidden_layers, lambda prefix: LlamaDecoderLayer(...))` at line 383 — this helper (from `vllm/model_executor/models/utils.py:685`) partitions the layers across PP ranks and fills the gaps with `PPMissingLayer` placeholders so parameter names still line up.
- `forward(input_ids, positions, intermediate_tensors, inputs_embeds=None, **extra_layer_kwargs)` at line 400:

```400:439:vllm/model_executor/models/llama.py
    def forward(self, input_ids, positions, intermediate_tensors,
                inputs_embeds=None, **extra_layer_kwargs):
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None \
                else self.embed_input_ids(input_ids)
            residual = None
        else:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual,
                                            **extra_layer_kwargs)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states,
                                        "residual": residual})

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
```

- `load_weights` at line 441:

```441:443:vllm/model_executor/models/llama.py
    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)
```

`AutoWeightsLoader` (`vllm/model_executor/models/utils.py`) walks the HF weight iterator, applies the `WeightsMapper`, resolves shard offsets for `MergedColumnParallelLinear` / `QKVParallelLinear` via `packed_modules_mapping`, and calls each parameter's `weight_loader`.

### 4e. LlamaForCausalLM

`llama.py:446` — the outer class. Bases: `LocalArgmaxMixin, nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3, SupportsQuant`.

Class attributes worth noting:

```457:464:vllm/model_executor/models/llama.py
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
```

`packed_modules_mapping` tells `AutoWeightsLoader` and LoRA which HF tensors were fused. `embedding_modules` tells the LoRA loader which parameters are embeddings.

Forward + compute_logits are trivial thin wrappers:

```516:533:vllm/model_executor/models/llama.py
    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        model_output = self.model(input_ids, positions, intermediate_tensors,
                                  inputs_embeds=inputs_embeds)
        return model_output

    def compute_logits(self, hidden_states):
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits
```

## 5. Diagram: LLaMA from top to bottom

```mermaid
flowchart TB
    LFCG[LlamaForCausalLM<br/>SupportsLoRA/PP/Eagle/Quant] --> LM[LlamaModel<br/>@support_torch_compile]
    LM --> EMB[VocabParallelEmbedding<br/>PP rank 0 only]
    LM --> LAYERS[layers 0..N-1<br/>via make_layers PP-partitioned]
    LAYERS --> LDL[LlamaDecoderLayer]
    LDL --> RN1[RMSNorm input_layernorm]
    LDL --> LA[LlamaAttention]
    LDL --> RN2[RMSNorm post_attn_layernorm]
    LDL --> MLP[LlamaMLP]
    LA --> QKV[QKVParallelLinear]
    LA --> ROT[Rotary via get_rope]
    LA --> ATT[Attention module<br/>attention/attention.py]
    LA --> OPR[RowParallelLinear o_proj]
    MLP --> GU[MergedColumnParallelLinear gate_up]
    MLP --> SIL[SiluAndMul]
    MLP --> DP[RowParallelLinear down]
    LM --> LN_F[RMSNorm final<br/>PP last rank]
    LFCG --> LMHEAD[ParallelLMHead<br/>PP last rank]
    LFCG --> LP[LogitsProcessor]
```

## 6. Comprehension checks

1. What is the difference between `packed_modules_mapping` and `hf_to_vllm_mapper`? Which one is used at LoRA time and which at weight-load time?
2. Where in `LlamaAttention.__init__` is the KV cache actually referenced? (Trick question — read the answer in `Attention.__init__` at `vllm/model_executor/layers/attention/attention.py:204`.)
3. If you added a new architecture `MyForCausalLM` and did not put it in `_VLLM_MODELS`, what would `ModelRegistry.resolve_model_cls` do? Walk through the `_try_resolve_transformers` fallback in `registry.py:1096`.
4. Why does `LlamaModel` use `@support_torch_compile` at the model level but `LlamaAttention` does not?
5. In `LlamaDecoderLayer.forward`, why does the first layer pass `residual=None` while the rest pass a residual tensor? What does `RMSNorm(hidden_states, residual)` do differently from `RMSNorm(hidden_states)`?

## 7. Hands-on exercise

Open `vllm/model_executor/models/llama.py` and answer these by reading the code (no running):

1. Which HF weight name gets loaded into `self.model.layers[3].self_attn.qkv_proj` for the "K" shard? Walk through the mapper transformation.
2. When TP=4 and the model has 32 attention heads, how many heads live on each rank? (Look at `LlamaAttention.__init__` lines 140–158.)
3. If PP=2 and the model has 32 hidden layers, which layer indices live on PP rank 0? (Look at `make_layers` in `vllm/model_executor/models/utils.py:685`, and `get_pp_indices`.)

Then predict what would break if you removed the `hf_to_vllm_mapper` in `LlamaModel`. Verify by reading `AutoWeightsLoader.load_weights`.

Tomorrow (Day 3): MoE models (Mixtral), multimodal (LLaVA), and non-transformer hybrids (Mamba / Jamba).
