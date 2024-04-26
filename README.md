# causalLM-pretraining-from-scratch

**This is a 5M parameter Causal Language model based on Transformer Decoder only architecture, which is trained on just 2.3M tokens.**


**Arcitecture details:**
1. Grouped Multi Query attention with 2:1 query, key ratio.
2. ROPE with theta of 1e5.
3. GELU activation with Tanh approximation.
4. RMS Norm(Non parameterised) applied for boh input and output.

**Model config:**

```config_dict = {
    "batch_size" : 56,
    "block_size" : 256,
    "embd" : 256,
    "n_heads" : 8,
    "n_kvheads" : 4,
    "head_size" : 256 // 8,
    "dropout" : 0.2,
    "n_layer" : 3,
    "learning_rate" : 3e-4
    "norm_eps" : 2e-4,
    "device" : 'cuda',
    "accumulation_steps" : 1,
    "epochs" : 1,
    "eval_iters" : 150,
    "start_pos" : 0,
    "temperature" : 0.5,
    "warmup_steps" : 3000
}


jnljj
