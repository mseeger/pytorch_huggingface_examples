# Work Items

Repos I am currently working on:
- `git@github.com:mseeger/transformers.git`
- `git@github.com:mseeger/litgpt.git`


## Done


## In Review

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

### LitGPT: Improvements of `KVCache`

`git/forks/litgpt`, branch `kvcache_improvements2`:

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

OK: https://github.com/Lightning-AI/litgpt/pull/1870


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
- Run existing tests, only for models changed [OK]
  run_tests, models_changed.txt
- Add test for gpt_neox model: Use the litgpt ones [OK]
- Write common Mixin [OK]
- Insert into all affected models, run tests [HIER]
  - aria: Hangs [TODO]
  - chameleon: Needs config.vocabulary_map [TODO]
  - clvp: ClvpRotaryPositionalEmbedding.forward is different [TODO]
  - codegen: Nonstandard, no *Embedding [TODO]
  - cohere [OK]
  - dbrx: Non-standard [TODO]
  - esm: What happens if position_ind is used in forward?? [TODO]
  - falcon [OK]
  - gemma: Tough [OK]
  - gemma2: [OK]
  - glm [OK]
  - gpt_neox, gpt_neox_japanese [OK]
  - gptj [HIER]

`run_tests`:
```bash
#!/bin/bash

for x in $(cat models_changed.txt)
do
  echo "[$x]"
  pytest tests/models/$x/ >$x.std.log 2>$x.err.log
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

Tests which failed:
- aria: cannot import name 'PILImageResampling' from 'transformers.image_utils'
  FIXED
- clvp: Errors [30] FIXED
- phi: Errors [1] FIXED

All tests pass now!