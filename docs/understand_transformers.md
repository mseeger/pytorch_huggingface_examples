# Learning about Modern Transformers by Studying LitGPT

The [LitGPT](https://github.com/Lightning-AI/litgpt/tree/main) codebase from
Lighting-AI provides clean, performant implementations of many current large
language models (LLMs).

By studying the code, we can learn about:
* The many variants of LLMs used today
* How these models are implemented in PyTorch
* What are the relevant config arguments for these models?

The idea is to briefly read up on techniques whenever they appear in the code,
unless they are highly specific.

Why not study Hugging Face?
* The LitGPT codebase is much simpler. Common concepts are factored out in a
  single code file
* LitGPT covers a much smaller range of models, but has the most important ones
  people use today
* Due to simplicity and modularity, it is simpler to modify and extend the
  LitGPT codebase


## Configuration

File: `litgpt/config.py`.

General size parameters:
- `block_size`: Context width. Maximum (and default) value for `max_seq_length`.
- `vocab_size`, `padded_vocab_size`: Number of different tokens in vocabulary.
  Latter padded to nearest multiple of `padding_multiple`.
- `n_layer`: Number of layers (or transformer blocks)
- `n_embd`: Embedding dimension. Each token is associated with vector of this size.

Transformer block (structure, normalizations):
- `norm_class`, `norm_class_name`, `norm_eps`: Type of normalization used at different
  places in transformer block (choices "LayerNorm", "RMSNorm")
- `shared_attention_norm`, `parallel_residual`: See [GPT(nn.Module)](#gptnnmodule).
- `post_attention_norm`, `post_mlp_norm`: See [GPT(nn.Module)](#gptnnmodule).

Transformer block (self-attention):
- `n_head`: Number of heads in multi-head attention (MHA).
- `head_size`: Dimension per head. Defaults to `n_embd // n_head`, but can be
  different.
- `n_query_groups`: Defaults to `n_head` for MHA. If smaller, must have
  `n_head % n_query_groups == 0`. In MHA, we use `n_head` Q, K, V vectors of
  size `head_size` each. If `n_query_groups < n_head`, some K, V vectors are shared.
  Namely, there are `n_query_groups` K, V vectors, and `n_head` Q vectors. Put
  differently, there are `n_query_groups` query groups, each consisting of one K,
  one V and `n_head / n_query_groups` Q vectors. This leads to smaller KV-caches.
  MQA (multi-query attention) has `n_query_groups = 1`.
- `attn_bias`: Whether linear blocks mapping to Q, K, V vectors have biases.
  What is used is `bias or attn_bias`.
- `sliding_window_size`, `sliding_window_layer_placing`: See
  [CausalSelfAttention(nn.Module)](#causalselfattentionnnmodule).
- `attention_scores_scalar`: Q, K inner product scores are scaled by
  `1 / sqrt(d)`. If given, `d = attention_scores_scalar`, otherwise (default)
  `d = head_size`.
- `attention_logit_softcapping`: Inner product scores going into softmax are
  softly capped to `[-attention_logit_softcapping, attention_logit_softcapping]`.
  **Note**: If this is used, cannot use
  `torch.nn.functional.scaled_dot_product_attention`, so no Flash attention,
  see [CausalSelfAttention(nn.Module)](#causalselfattentionnnmodule). This can
  be a lot slower and require more memory. Only used in Gemma-2 models.

Transformer block (MLP):
- `mlp_class`, `mlp_class_name`: Type of MLP being used, choices are
  "GptNeoxMLP", "LLaMAMLP", "GemmaMLP", "LLaMAMoE", defaults to
  "GptNeoxMLP"
- `intermediate_size`: Size of hidden layer in MLP. Typically a small multiple
  of `n_embd`
- `gelu_approximate`: Parameter for `torch.nn.functional.gelu` (for those MLPs
  using GELU)
- `bias`: Whether linear blocks in MLP have biases. Also affects linear blocks
  in self-attention.
- `n_expert`, `n_expert_per_token`: Parameters of `LLaMAMoE` (special MoE model).

GPT before/after blocks:
- `scale_embeddings`: Input embeddings scaled by `sqrt(n_embd)`
- `lm_head_bias`: Linear head of GPT has biases
- `final_logit_softcapping`: Final outputs after `lm_head` softly capped to
  `[-final_logit_softcapping, final_logit_softcapping]`

RoPE parameters: See [Rotary Position Embedding (RoPE)](#rotary-position-embedding-rope)
for parameter semantics:
- `rope_base`: Defaults to 10000 (from paper).
- `rotary_percentage`: Defaults to 0.25.
- `rope_condense_ratio`: Defaults to 1.
- `rope_adjustments`: Dictionary with further parameters (optional).


## Model (GPT)

File: `litgpt/model.py`.

### GPT(nn.Module)

- Input embeddings `transformer.wte`, blocks `transformer.h` (`config.n_layer`
  `Block` objects), final function `transformer.ln_f` (norm of
  `config.norm_class`)
- Linear head `lm_head`, final dimension from `config.n_embd` to
  `config.padded_vocab_size`. Maps to final logits over padded vocabulary
- Maintains `cos`, `sin` (RoPE cache)
- `max_seq_length`: Must be `<= config.block_size` (the initial value), can be
  changed to save time and memory.
- `forward`: If `input_pos` given, this is generative inference for token at this
  position, so the KV cache is used. Note that `mask` as argument to the blocks
  is only used in this case. The default `mask=None` is used for causal
  self-attention.
- Optional (`config.scale_embeddings`): Scaling input embeddings
- Optional (`config.final_logit_softcapping`): Soft capping of final outputs

Soft capping works as follows:
```python
def do_softcapping(x: torch.Tensor, thresh: float) -> torch.Tensor:
    return torch.tanh(x / thresh) * thresh
```

### Block(nn.Module)

This code is my modification, it fixes a bug (`self.post_mlp_norm` not used
in the parallel residuals case):
```python
def forward(
    self,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    input_pos: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x_normed = self.norm_1(x)
    attention_output = self.attn(x_normed, cos, sin, mask, input_pos)
    attention_output = self.post_attention_norm(attention_output)

    if self.config.parallel_residual:
        if not self.config.shared_attention_norm:
            x_normed = self.norm_2(x)
        x = attention_output + x
    else:
        x = attention_output + x
        x_normed = self.norm_2(x)
    return self.post_mlp_norm(self.mlp(x_normed)) + x
```

- Main parts: `attn` (`CausalSelfAttention`), `mlp` (`config.mlp_class`).
  Depending on the model, the MLP is different
- Normalization before `attn`, `mlp`, optionally after as well. All norms of type
  `config.norm_class`
- Default is non-parallel residuals
- Parallel residuals: `attn`, `mlp` in parallel and added. Optionally, the "before"
  normalizations can be shared. Note that none of the models in `config.py` are using
  parallel residuals. Is this a good idea?


### CausalSelfAttention(nn.Module)

- `attn`: `Linear(n_embd, shape)`, where `shape = n_head + 2 * n_query_groups) * head_size`
- `proj`: `Linear(n_head * head_size, n_embd)`
- `kv_cache`: TODO
- `apply_sliding_window_attention`: If active, and `block_idx` condition
  (TODO)

If `q_per_kv = n_head // n_query_groups` and `total_qkv = q_per_kv + 2`, each
query group has `total_qkv` vectors. For default MHA, `q_per_kv = 1, total_qkv = 3`.

```python
def forward(
    self,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    input_pos: Optional[torch.Tensor] = None,
) -> torch.Tensor:
```
`x` has shape `(B, T, n_embd)`. `input_pos` is given for generative inference.
Let `hs = head_size`, `nqg = n_query_groups`:

- Apply `attn`: `(B, T, total_qkv * nqg * hs)`
- Reshape and permute: `(B, nqg, total_qkv, T, hs)`
- Split into `q` of shape `(B, nqg, q_per_kv, T, hs)`, `k`, `v` of shape
  `(B, nqg, 1, T, hs)`
- If not MHA and (not generative inference or not MQA): Expand `k`, `v` to
  same shape `(B, nqg, q_per_kv, T, hs)` as `q`. This means the
  singleton dimension is expanded by repeating (broadcasting), just using a
  stride of 0 (read-only!).
- Reshape all three to `(B, n_head, T, hs)`
- RoPE: Apply `cos`, `sin` to `q`, `k`
- If generative inference: This is correct for `q`, but not for `k`, `v`.
  Get them from `kv_cache`. Then, `k`, `v` have shape `(B, n_head, T2, hs)`
  with `T2 >= T`.
- If `apply_sliding_window_attention`: Add something to `mask` (TODO)
- `y = self.scaled_dot_product_attention(q, k, v, mask)`, result has shape
  `(B, T, n_head, hs)`.
- Reshape to `(B, T, hs * n_head)`, then apply `proj`

```python
def scaled_dot_product_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
```
`q` has shape `(B, n_head, T1, hs)`, `k`, `v` have shape `(B, n_head, T2, hs)`, 
where `T2 >= T1`. If `mask` is given, it has shape `(T1, T2)`. Note that the
default is `mask = None`, which means causal self-attention, where also
`T1 == T2`.

- The output is a tensor of shape `(B, T1, n_head, hs)`. The computations
  deliver `(B, n_head, T1, hs)`, then a final `transpose(1, 2)`.
- If `config.attention_logit_softcapping` is not given, this simply calls
  `torch.nn.functional.scaled_dot_product_attention`, which is highly
  optimized (Flash attention). We pass `attn_mask=mask`, `is_causal=mask is None`,
  and no dropout. The default is causal with `mask=None`, a mask is used
  only for generative inference.
- If `config.attention_logit_softcapping`, the Q-K inner product score values
  are soft-capped before going into the softmax. This can be a lot slower and
  materializes a `(B, n_head, T1, T2)` tensor.


### GptNeoxMLP, LLaMAMLP, GemmaMLP

These are different implementations of the MLP (or FFN) part of a transformer
block. Here, `fc` or `fc_?` are `Linear(n_embd, intermediate_size)` for hidden
layer, and `proj` is `Linear(intermediate_size, n_embd)` for output layer.

- `GptNeoxMLP` (default). `fc`; then `gelu(x_fc)`
  (which is `x_fc * cdf_gauss(x_fc)`, using a `tanh` approximation if
  `gelu_approximate == "tanh"`; then `proj`.
- `LLaMAMLP`. `fc_1` and `fc_2`; then `silu(x_fc_1) * x_fc_2`, where
  `silu(x) = x * sigmoid(x)`; then `proj`.
- `GemmaMLP`. `fc_1` and `fc_2`; then `gelu(x_fc_1) * x_fc_2`; then `proj`.

There is also `LLaMAMoE`, which ensembles `config.n_expert` `LLaMAMLP`
networks.


### Rotary Position Embedding (RoPE)

Paper: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v5)

Where in `litgpt/model.py`:
- Functions `apply_rope`, `build_rope_cache`
- `GPT`: Compute `sin`, `cos` with `self.rope_cache`, `build_rope_cache`,
  depends on `max_seq_length`
- `CausalSelfAttention`: Apply RoPE (`cos`, `sin`) to `q` and `k`, using
  `apply_rope` (in the same way)

RoPE is applied to the K, V vectors before they go into the inner product
attention. This happens in MHA in every block, as a form of *relative position
encoding*, in that the score between a key and a query vector decays with the
distance of the respective tokens in the sequence, but does not depend on their
abnsolute positions. In contrast, the original transformer used *absolute position
encoding*, in that positional encodings were added to input embeddings at
the very start, and transformer blocks did not depend on positional encoding.
RoPE and related techniques are crucial if models are pretrained with a certain
context width, which is then lifted to a much larger size by a bit of
fine-tuning.

Parameters:
- `rope_base`: Base for computing the `theta` values. Defaults to 10000 (from
  paper), but is related to context width (so is much larger for some of the
  models).
- `rotary_percentage`: Defaults to 0.25, but quite a few models use 1. Initial
  fraction of Q, K vectors (total size `head_size`) to which RoPE is applied.
  The rest is not modified. Note that `rope_n_elem = int(rotary_percentage * head_size)`
  is the number of initial elements to which RoPE is applied.
- `rope_condense_ratio`: Defaults to 1, different only for one model.
- `rope_adjustments`: Dictionary with further parameters (optional). These are
  used only in the LlaMA 3.1 and 3.2 models.

The *rope cache* consists of two tensors `cos`, `sin` of shape
`(max_seq_length, rope_n_elem)`. Ignoring `rope_adjustments`:

```python
theta = 1.0 / (base ** (torch.arange(0, rope_n_elem, 2).float() / rope_n_elem))
seq_idx = torch.arange(max_seq_length) / rope_condense_ratio
idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)
cos = torch.cos(idx_theta)
sin = torch.sin(idx_theta)
```

This means that row `i` of `cos` is given by the concatenation of two copies
of `[cos(i * theta[0]), cos(i * theta[1]), ..., cos(i * theta[-1])]`.

If `rope_adjustments` is given, the `theta` values are adjusted in a certain
way, this is not part of the original RoPE paper. It is used in Llama-3.1,
Llama-3.2, Llama-3.3 models.

RoPE is then applied to the first `rope_n_elem` entries of K and Q vectors. This
is done like in the RoPE paper, except there they group scalars `(0, 1), (2, 3), ...`
for rotation, whereas here they group `(0, ne2), (1, ne2 + 1), ...`, where
`ne2 = rope_n_elem / 2`. This difference could be a problem. If a model was
pretrained with the RoPE paper convention, then fine-tuning with the different
convention here would be suboptimal.

#### What is Hugging Face doing?

`src/transformers/modeling_rope_utils.py`:
- `_compute_default_rope_parameters`, `ROPE_INIT_FUNCTIONS["default"]`: Compute
  `theta` parameters by default.
- `_compute_llama3_parameters`, `ROPE_INIT_FUNCTIONS["llama3"]`: Additional
  adjustments given by `rope_adjustments` above.
- They seem to only register the `theta` vector, as "inv_freq"

The rest is done specific for every model. For example:
`src/transformers/models/gpt_neox/modeling_gpt_neox.py`:
- `GPTNeoXRotaryEmbedding`: Registers `theta` as "inv_freq". `forward` computes
  `sin`, `cos` of shape `(B, T, hs)`, where `B` is batch size. The input is
  `position_ids` of shape `(B, T)`, which can have different positions for each
  case in the batch. If `B = 1` and `position_ids = arange(0, T)`, this is the
  same as in LitGPT.
- `apply_rotary_pos_emb`, `rotate_half`: They do the same as in LitGPT.

Hugging Face does the same as LitGPT, which deviates from the RoPE paper.

**But**: The Llama reference implementation groups scalars as `(0, 1), (2, 3), ...`.
Namely, in https://github.com/meta-llama/llama/blob/main/llama/model.py#L132:

```python
xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
```

Here, the reshaping of `xq`, `xk` means that the final dimension `(hs,)` is
reshaped to `(hs/2, 2)`, inducing a grouping `(0, 1), (2, 3), ...`.
This reference implementation is also used in TorchTune, see
https://pytorch.org/torchtune/0.3/generated/torchtune.modules.RotaryPositionalEmbeddings.html.

This has been noted: https://github.com/huggingface/transformers/issues/25199.
Hugging Face corrects for this by permuting the K and V weight linear transform
weights accordingly when loading Llama checkpoints. It is not very clear why
there is this difference.


### Ignored For Now

- Sliding window attention in `CausalSelfAttention`, config parameters
  `sliding_window_size`, `sliding_window_layer_placing`. Used in Gemma-2
  (size 4096, every even layer), Phi-3-mini (size depends on context
  width, every layer), Mathstral, Mistral.
- `LLaMAMoE` (`config.mlp_class_name`). More expensive MoE MLP part. Used in
  Mixtral models only.


## Generative Inference, KV Cache

If `input_pos` is given in `GPT.forward`, it contains the token positions
corresponding to the token indices in `idx`, where `idx.shape == (B, idx_size)`.
Either `input_pos.shape == (idx_size,)` or `input_pos.shape == (B, idx_size)`
(batched index). If `input_pos is None`, this corresponds to the default case
`input_pos == arange(idx_size)`, when all inputs up to `idx_size` are given.

The case `input_pos is not None` is needed for generative inference. Typically,
tokens are sampled one by one, after the prompt has been processed. Then,
`idx_size == 1`. Also, a very large prompt can be processed in batches of size
`idx_size > 1`, updating the KV cache in between.

In the generative inference case, `q`, `k`, `v` in
`CausalSelfAttention.forward` are computed for positions in `input_pos`. This
is correct for `q`, but not for `k`, `v`, where we need vectors for earlier
positions as well. This is why we have there:

```python
if input_pos is not None:
    if not isinstance(self.kv_cache, KVCache):
        raise TypeError("You need to call `gpt.set_kv_cache()`")
    k, v = self.kv_cache(input_pos, k, v)
```

The resulting `k`, `v` are extended by the K and V vectors from the cache, so
that causal self-attention is correct in this case. At the same time, the KV
cache is updated by the new `k`, `v` vectors passed as input.

HIER: How does the implemented KV cache work? Are there other implementations?
- `GPT`: `set_kv_cache`, `clear_kv_cache`
- `CausalSelfAttention`: `build_kv_cache`, `kv_cache` member
- `KVCache` class. Don't see other implementations

Idea for project: A performant KV cache implementing eviction strategies like
heavy hitter oracle.


## Fine-tuning with LoRA and Adapters 

HIER!
