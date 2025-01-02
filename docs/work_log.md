# Work Items

Repos I am currently working on:
- `git@github.com:mseeger/transformers.git`
- `git@github.com:mseeger/litgpt.git`


## Done

### LitGPT: Small fixes and refactoring

`git/forks/litgpt`, branch `small_fixes`:

- `config.sliding_window_layer_period` instead of overwriting
  `config.sliding_window_layer_placing` (str to int)
- Function `do_softcapping`
- Comment on `config.attention_logit_softcapping`: Will hit performance if this is used
- `cos`, `sin` passed down in `GPT.forward` must have batch dimension if
  `input_pos is None`. This probably worked because PyTorch is adding singleton
  dimensions at the front of a shape for broadcasting, but it did confuse me.
- `Block.forward`: Missed `self.post_mlp_norm` if `parallel_residual`

OK: https://github.com/Lightning-AI/litgpt/pull/1861

### PyTorch Sources of Multi-Head Attention

We know that `torch.nn.functional.scaled_dot_product_attention` computes the
relevant part of MHA in the most efficient way, but it does not return the
attention weights. We need the latter for computing scores related to KV
caching, such as H2O. Here, we dig into the sources to see how this could be
done.

* [_scaled_dot_product_attention_math](https://github.com/pytorch/pytorch/blob/2fa09853cbd5c774262a843436347fa14c1012a0/aten/src/ATen/native/transformers/attention.cpp#L792):
  This `C++` function returns a tuple `(attn_output, attn_weights)`. We would
  need the second entry.
* [jagged_dot_product_attention](https://github.com/pytorch/pytorch/blob/2fa09853cbd5c774262a843436347fa14c1012a0/torch/nested/_internal/sdpa.py#L678):
  Calls `_scaled_dot_product_attention_math` if `backend_choice == SDPBackend.MATH`.
  There are other branches as well, calling functions from `torch.ops.aten`. Where
  is the code for these?
* [aten/src/ATen/native/transformers/attention.cpp](https://github.com/pytorch/pytorch/blob/2fa09853cbd5c774262a843436347fa14c1012a0/aten/src/ATen/native/transformers/attention.cpp#L696):
  `scaled_dot_product_attention`: Depending on `backend`, many different functions
  are called. All of them return a tuple `(output, logsumexp)`, and the function
  returns the first argument. Subfunctions:
  - `at::_scaled_dot_product_cudnn_attention`:
  - `at::_scaled_dot_product_flash_attention`:
  - `at::_scaled_dot_product_flash_attention_for_cpu`:
  - `at::_scaled_dot_product_efficient_attention`:
  - `at::_scaled_dot_product_fused_attention_overrideable`: Not implemented
  - `at::_scaled_dot_product_attention_math_for_mps`:
  - `at::_scaled_dot_product_attention_math`: Returns `(attn_output, attn_weights)`
* [aten/src/ATen/native/transformers/cuda/attention.cu](https://github.com/pytorch/pytorch/blob/2fa09853cbd5c774262a843436347fa14c1012a0/aten/src/ATen/native/transformers/cuda/attention.cu):
  In this code, if `query` has shape `(B, M, num_heads, K)`, `key` has shape
  `(B, N, num_heads, K)`, `value` has shape `(B, N, num_heads, Kv)`, the functions
  return a tuple `(output, logsumexp)`, where `output` has shape `(B, M, num_heads, Kv)`
  and `logsumexp` has shape `(B, num_heads, max_seqlen_q)`, where by default
  `max_seqlen_q = M`. `logsumexp` cannot be `attn_weights`, they would have to
  have a shape depending on `max_seqlen_k` or `N`. DAMN!

**Note**: We only need the attention weights in the generative inference case,
when `query` is small (usually a single token). In this case, a naive
implementation of MHA suffices. There is no need to call
`torch.nn.functional.scaled_dot_product_attention`.

### LitGPT: Improvements of `KVCache`

`git/forks/litgpt`, branch `kvcache_improvements4`:

- `CausalSelfAttention.forward`: Move expand and reshape to after `kv_cache` is
  used. Ensures that KV cache size depends on `n_query_groups`. [OK]
- 'KVCache': `covering_length`, modify `forward`, and subselect `mask` in
  `GPT.forward`. Ensure that MHA is done with shorter `k`, `v`. [OK]
- Refactor `GPT.forward` to simplify `adapter.py` [OK]
- Refactor `adapter.py`, `adapter_v2.py`, `lora.py` to use as much code of
  `model.py` as possible. Right now, this is copy&paste [OK]
- Make test_against_gpt_neox_model work: This is due to a bug in
  Hugging Face! [OK]
- Cannot compute covering_length from input_pos, this induces a graph break.
  But can pass this value in, which works in the normal cases. If not passed
  in, the subselection is not done [OK]
- Make test_model_compile work [OK]
- Run all tests [MOVE ON]
  tests
  RuntimeError: Command 'litgpt finetune_lora checkpoints
  Error:
  usage: litgpt [options] finetune_lora [-h] ...
  error: cannot unpack non-iterable ActionTypeHint object
  - Also in main branch
  - ActionTypeHint is in jsonargparse
- Check other generate code (input_pos given), insert input_pos_maxp1 if possible.
  There is generate.base.batched_generate_fn, but this is not currently
  used [OK]
- New unit tests [OK]
- Reproduce HF bug? If so, file bug report [OK]
- Rebase on current main [OK]
- Restore old copy&paste constructors: [OK]
  Branch: `kvcache_improvements4`
- Fix tests [OK]
  test_adapter_v2.py still fails. Why?

OK: https://github.com/Lightning-AI/litgpt/pull/1891


## In Review



## Currently Working On

### Hugging Face transformers: Fix RoPE Bug

`git/forks/transformers`, branch `fix_rope`.

Code to reproduce the bug: Add this to `src/transformers/models/gpt_neox/modeling_gpt_neox.py`:
```python
def reproduce_bug():
    # Then:
    # head_size = hidden_size 
    # rotary_ndims = int(head_size * rotary_pct) = 3
    config = GPTNeoXConfig(
        vocab_size=96,
        max_position_embeddings=32,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=8,
        intermediate_size=3 * 32,
        rotary_pct=0.75,
        use_parallel_residual=False,
    )
    model = GPTNeoXModel(config)
    input_ids = torch.randint(0, config.vocab_size, (1, config.max_position_embeddings))
    logits = model(input_ids)
    print(f"logits.shape = {logits.shape}")


if __name__ == "__main__":
    reproduce_bug()
```

Any models on hub affected?
- gpt-neox-20b:
  hidden_size=6144, num_attention_heads=64, rotary_pct=0.25:
  rotary_ndims=24
- japanese-gpt-neox-3.6b-instruction-sft:
  hidden_size=2816, num_attention_heads=22, rotary_pct=1:
  rotary_ndims=128
- gpt-neox-japanese-2.7b:
  hidden_size=2560, num_attention_heads=32, rotary_pct=1:
  rotary_ndims=80
- gpt_neox_225M:
  hidden_size=1024, num_attention_heads=12, rotary_pct=0.25:
  head_size=85, rotary_ndims=21 [UUPS!]
- tiny-random-GPTNeoXForCausalLM:
  hidden_size=32, num_attention_heads=4, rotary_pct=0.25:
  rotary_ndims=2
- shahules786
  hidden_size=1024, num_attention_heads=16, rotary_pct=0.25:
  rotary_ndims=16
- mkshing
  hidden_size=768, num_attention_heads=12, rotary_pct=0.25:
  rotary_ndims=16
- mkshing
  hidden_size=1024, num_attention_heads=12, rotary_pct=0.25:
  head_size=85, rotary_ndims=21 [UUPS!]

Any others under `models` use `ROPE_INIT_FUNCTIONS`?
- gpt_neox
  HAS rotary_ndims
  - GPTNeoXAttention.rotary_ndims
    GPTNeoXAttention._attn_projections_and_rope
  - GPTNeoXRotaryEmbedding: self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
    GPTNeoXRotaryEmbedding.forward: Compute cos, sin [should be subselected here]
  - apply_rotary_pos_emb
  - GPTNeoXModel.forward: position_embeddings = self.rotary_emb(hidden_states, position_ids)

UURGH!
- Some use config.partial_rotary_factor, others use config.rotary_pct
- But config.partial_rotary_factor is always there:
  Make sure rotary_pct not used in code

- aria
  Has no config.rotary_pct (always 1)
  Seems derived from llama model
- cohere
  Has no config.rotary_pct (always 1)
- falcon
- gpt_neox_japanese
  HAS rotary_ndims
- granite
- granitemoe
- llama
- mllama
- nemotron
  Has config.partial_rotary_factor, but not implemented (??)
- olmo
- olmoe
- persimmon
  HAS rotary_ndims
- phi
  HAS rotary_ndims
- phimoe
- qwen2
- qwen2_moe
- qwen2_vl
- stablelm
  HAS rotary_ndims
- starcoder2
  But seems derived from qwen2

Which has `apply_rotary_pos_emb`?
- aria*
- chameleon*
- clvp [only encoder!]
- codegen
- cohere*
- dbrx*
- esm
- falcon*
- gemma*
- gemma2*
- glm* [partial_rotary_factor]
- gpt_neox* [rotary_pct]
- gpt_neox_japanese* [rotary_pct]
- gptj
- granite*
- granitemoe*
- idefics*
- jetmoe*
- llama*
- mimi*
- mistral*
- mixtral*
- mllama*
- moshi*
- nemotron* [partial_rotary_factor]
- olmo*
- olmo2*
- olmoe*
- persimmon* [partial_rotary_factor]
- phi* [partial_rotary_factor]
- phi3*
- phimoe*
- pixtral
- qwen2*
- qwen2_moe*
- qwen2_vl*
- recurrent_gemma* [partial_rotary_factor]
- stablelm* [partial_rotary_factor]
- starcoder2*

`*` means that `self.rotary_emb = ...Embedding(...)` is used in the model, so
that the default `embedding_from_model` should work.

TODO:
- Check which code is affected [OK]
  Also cover against head_size being odd, this is not forbidden
- nemotron: implement! Fix. [OK]
- Allow for partial_rotary_factor as alias of rotary_pct in gpt_neox
  configs
- recurrent_gemma: rot and pass. Changed [OK]
- Refactor models one by one [OK]
- Fixed several bugs in gptj/modeling_gptj.py, GPTJAttention: [OK]
  - pos_embd_dim is self.head_dim if config.rotary_dim is None!
  - RoPE is applied separately to every head. The flatten(-2) in
    rotate_every_two is wrong!
- Fix in olmo2: Attention classes Olmo2* were not used, at least
  in modular_olmo2
- Cleaned up recurrent_gemma
- Fixed bug in esm: RoPE embedding depends on position_ids, must be passed
- Run existing tests, only for models changed [OK]
  run_tests, models_changed.txt
- Add test for gpt_neox model: Use the litgpt ones [OK]
- Write common Mixin [OK]
- Insert into all affected models, run tests:
  - aria: Text model is just llama [DROP]
  - chameleon: Needs config.vocabulary_map [OK]
  - clvp: Weird case, drop this [DROP]
  - codegen: Makes no sense [DROP]
  - cohere [OK]
  - dbrx: Non-standard [OK]
  - esm: Had to fix a bug [OK]
  - falcon [OK]
  - gemma: Tough [OK]
  - gemma2: [OK]
  - glm [OK]
  - gpt_neox, gpt_neox_japanese [OK]
  - gptj: Nonstandard! Tough! [OK]
  - granite: [OK]
  - granitemoe [OK]
  - idefics: Mixed up with image model [DROP]
  - jetmoe: Nonstandard [OK]
  - llama [OK]
  - mimi: Nonstandard [OK]
  - mistral [OK]
  - mixtral [OK]
  - mllama [OK]
  - nemotron [OK]
  - olmo [OK]
  - olmo2: [OK]
  - olmoe [OK]
  - persimmon [OK]
  - phi [OK]
  - phi3 [OK]
  - phimoe [OK]
  - pixtral: No changes [OK]
  - qwen2 [OK]
  - qwen2_moe [OK]
  - qwen2_vl: rope_scaling["mrope_section"] not documented [DROP]
  - recurrent_gemma [OK]
  - stablelm [OK]
  - starcoder2 [OK]

- Run all new tests once more: [OK]
  - falcon
- Look for _gradient_checkpointing_func: Ordering of args!
  All but clvp, esm, qwen2_vl [OK]
- Check modeling for modular <-> modeling [OK]
  glm, starcoder2, aria, olmo2
- Pull recent main, rebase [OK]
- Make Github properly work here (keys?), then push [OK]
- Run all new tests again [OK]
- Install torch-vision, run mllama test again [OK]
- First submit a PR with bug fixes: git diff main, what is not related to
  original job? [OK]
- Rebase to `fixrope_part1`: New branch `fix_rope3`

First PR: Branch `fixrope_part1`:
- Create branch by diff [OK]
- Run tests for all models [OK]
- Prepare: [OK]
  https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request
  Currently: make fixup [differences between modeling_* and generated]
- All sorts of CI failures: Run everything again! [OK]
  - Run tests [OK]
  - Run "make fixup" [OK]
- Check modeling_gptj changes, simplify [OK]
- Extend gptj changes to tf, flax as well [OK]
- Same for esm [OK]
- Fix codegen same as gptj [OK]
- Rebase to current main [OK]
- Comparison tests fail for gptj [HIER!]
  - Shapes are the same
  - OK: tests_torch_and_tf pass now!
  - tests_torch_and_flax fails now

OK: https://github.com/huggingface/transformers/pull/35376

Wait until this is in [HIER]

`run_tests`:
```bash
#!/bin/bash

for x in $(cat models_changed.txt)
do
  echo "[$x]"
  pytest -k rotary_ndims_odd tests/models/$x/ >$x.std.log
done
```
models_changed.txt:
```text
aria
chameleon
clvp
codegen
cohere
dbrx
esm
falcon
gemma
gemma2
glm
gpt_neox
gpt_neox_japanese
gptj
gptj
granite
granitemoe
idefics
jetmoe
llama
mimi
mistral
mixtral
mllama
moshi
nemotron
olmo
olmo2
olmoe
persimmon
phi
phi3
phimoe
pixtral
qwen2
qwen2_moe
qwen2_vl
recurrent_gemma
stablelm
starcoder2
```

### LitGPT: New abstraction for KVCache

Would like to implement a flexible KV cacheing framework on top of `LitGPT`.
This should mostly be done in a separate repository, but needs to be supported
on their side, mostly by passing attention weights in MHA to the KV cache.
To get started, we implement things in a branch of a fork. Later, we will
separate code into (a) own repo and (b) PR to them.

The first milestone would be to implement something that supports H2O.

Note that we can use the naive MHA implementation given in
`litgpt.model.CausalSelfAttention.scaled_dot_product_attention` (the branch
called if `config.attention_logit_softcapping` is given), because the `q`
tensor is small in this case.

The very first step is a proper design, at least for the `KVCache` API.

`git/forks/litgpt`, branch `kvcache`:

- Work out design in technical report [OK]
- Base class `KVCache` and `DenseKVCache` [OK]
- Class `AttnWeightsKVCache`: Generic code, except for `_update`  [OK]
- Implement class `H2OKVCache` [HIER]
- Change stuff in `model.py` to use new cache
- Implement single seq and batched generation loops
