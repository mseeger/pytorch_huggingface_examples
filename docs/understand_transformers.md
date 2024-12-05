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
  Latter padded to nearest multiple of `padding_multiple`, so that embedding
  parameters are power of 2.
- `n_layer`: Number of layers (or transformer blocks)
- `n_embd`: Embedding dimension. Each token is associated with vector of this size.

Transformer block (normalizations):
- `norm_class`, `norm_class_name`, `norm_eps`: Type of normalization used at different
  places in transformer block (choices "LayerNorm", "RMSNorm")
- `shared_attention_norm`, `parallel_residual`: See [GPT(nn.Module)](#gptnnmodule).
- `post_attention_norm`, `post_mlp_norm`: See [GPT(nn.Module)](#gptnnmodule).

Transformer block (self-attention):
- `n_head`: Number of heads in multi-head attention (MHA).
- `head_size`: Dimension per head. Defaults to `n_embd // n_head`, but can be
  different.
- `n_query_groups`: Defaults to n_head for MHA. If smaller, must have
  `n_head % n_query_groups == 0`. In MHA, we use `n_head` Q, K, V vectors of
  size `head_size` each. If `n_query_groups < n_head`, some K, V vectors are shared.
  Namely, there are `n_query_groups` K, V vectors, and `n_head` Q vectors. This
  leads to smaller KV-caches. MQA (multi-query attention) has `n_query_groups = 1`.
- `attn_bias`: Whether linear block mapping to Q, K, V vectors has biases.
  What is used is `bias or attn_bias`.
- `sliding_window_size`, `sliding_window_layer_placing`: See
  [CausalSelfAttention(nn.Module)](#causalselfattentionnnmodule).
- `attention_scores_scalar`: Q, K inner product scores are scaled by
  `1 / sqrt(d)`. If given, `d = attention_scores_scalar`, otherwise (default)
  `d = head_size`.
- `attention_logit_softcapping`: Inner product scores going into softmax are
  softly capped to `[-attention_logit_softcapping, attention_logit_softcapping]`.
  Note: If this is used, cannot use
  `torch.nn.functional.scaled_dot_product_attention`, so no Flash attention.

Transformer block (MLP):
- `mlp_class`, `mlp_class_name`: Type of MLP being used
- `intermediate_size`: Size of hidden layer in MLP
- `gelu_approximate`: Parameter for `torch.nn.functional.gelu`
- `bias`: Whether linear blocks in MLP have biases. Also affects
  projection block in self-attention.
- `n_expert`, `n_expert_per_token`: Parameters of `LLaMAMoE` (special MoE model).

GPT before/after blocks:
- `scale_embeddings`: Input embeddings scaled by `sqrt(n_embd)`
- `lm_head_bias`: Linear head of GPT has biases
- `final_logit_softcapping`: Final outputs after `lm_head` softly capped to
  `[-final_logit_softcapping, final_logit_softcapping]`

RoPE parameters:
- `rotary_percentage`: TODO
- `rope_condense_ratio`: TODO
- `rope_base`: TODO
- `rope_adjustments`: TODO


## Model (GPT)

File: `litgpt/model.py`.

### GPT(nn.Module)

- Input embeddings `transformer.wte`, blocks `transformer.h` (`config.n_layer`
  `Block` objects), final function `transformer.ln_f` (norm of
  `config.norm_class`)
- Linear head `lm_head`, final dimension from `config.n_embd` to `config.padded_vocab_size`
- Maintains `cos`, `sin` (RoPE cache)
- `max_seq_length`: Must be `<= config.block_size` (the initial value), can be
  changed to save time and memory.
- `forward`: If `input_pos` given, this is generative inference for token at this
  position, so the KV cache is used
- Optional (`config.scale_embeddings`): Scaling input embeddings
- Optional (`config.final_logit_softcapping`): Soft capping of final outputs

TODO:
- `rope_cache` (RoPE), `set_kv_cache` (KV cache)

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
  norms can be shared


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
- Split into `q` `(B, nqg, q_per_kv, T, hs)`, `k`, `v` `(B, nqg, 1, T, hs)`
- If not MHA and (not generative inference or not MQA): Expand `k`, `v` to
  same shape `(B, nqg, q_per_kv, T, hs)` as `q`. This means the
  singleton dimension is expanded by repeating (broadcasting), just using a
  stride of 0 (read-only!).
- Reshape all three to `(B, -1, T, hs)`
- RoPE: Apply `cos`, `sin` to `q`, `k` (TODO)
- If generative inference: Have `T == 1`, this is correct for `q`, but not
  for `k`, `v`. Get them from `kv_cache` (TODO)
- If `apply_sliding_window_attention`: Add something to `mask` (TODO)
- `y = self.scaled_dot_product_attention(q, k, v, mask)`
- Reshape to `(B, T, hs * n_head)`, then apply `proj`

TODO:
- RoPE
- Sliding window attention
- KV cache part