# causalLM-pretraining-from-scratch

**A 5M parameter Causal Language model based on Transformer Decoder only architecture, which is trained on just 2.3M tokens on an RTX 3060 consumer GPU.**

**Arcitecture details:**
1. Transformer - Decoder Only Arcitecture.
2. Grouped Multi Query attention with 2:1 query, key ratio.
3. ROPE with theta of 1e5.
4. GELU activation with Tanh approximation.
5. RMS Norm(Non parameterised) applied for both inputs and outputs.

**Model config:**

```
config_dict = {
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
```

**Datasets:** Harrypotter & LOTR complete series.

**Training details:**

1. The model was trained on the dataset of 2.3 million raw text tokens, with the block_size of 256 tokens and embedding dimensions of 256.
2. Used RoPE to extend contextual understanding during inference, enhancing token relations through relative position embeddings.
3. Used GELU activation function in the Feed forward layer as it works better(But nobody knows the exact reason).
4. Implemented Grouped Multi Query attention for faster training. But I didn't see much difference in the training time as my model is too small.
5. I've chosen RMS Normalization because it simplifies the process by removing extra mean calculations, which has no impact, as mentioned in the paper.
6. Normalized both the Inputs and Outputs of the model for better numerical stability and to avoid saddle points during the gradient optimization.
7. I used the AdamW optimizer with a weight decay of 0.99. Although a higher value slows down model convergence, Still I used it for minimizing the overfitting as much as possible.
8. I trained this model in two phases using two different learning schedulers. In the first phase, I used the CyclicLR scheduler, halving the amplitude scaling for each cycle, and in the second phase, I used the  Cosine scheduler with linear warmup. Although AdamW ideally pairs with the linear warmup cosine scheduler, I chose CyclicLR because it effectively explores the parameter space and improves the training convergence.
9. I trained this model using full precision floating point numbers because bfloat16 made the training slower for some reason. I also tried mixed precision training with PyTorch's AMP package, but it caused the numercial instability in the gradients.
10. At first, the gradient and parameters was high while training. Later on, it went down gradually and got closer to 1.

The overall training process went smoothly, and the final model effectively captures data patterns, sentence structures and some coherent sentences despite its small size. However, the model could be enhanced by adding more data and parameters to generate even more coherent sentences. As expected, the model was slightly overfitted, indicating there are no bugs in the model architecture and training code. This can be scaling to more parameters with minor modifications to the architecture.

