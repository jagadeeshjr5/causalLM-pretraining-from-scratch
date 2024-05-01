# causalLM-pretraining-from-scratch

**This is a 5M parameter Causal Language model based on Transformer Decoder only architecture trained on just >4M raw text tokens on an RTX 3060 gaming GPU.**

**Arcitecture details:**
1. Transformer - Decoder Only Arcitecture.
2. Grouped Multi Query attention with 2:1 query, key ratio.
3. ROPE with theta of 1e5.
4. GELU activation with tanh approximation.
5. RMS Norm(Non parameterised) applied for both inputs and outputs.
6. Remaining vanilla transformer stuff.

I've trained a custom BPE tokenizer with GPT-4 initial token splitter using Andrej Karpathy's minbpe(slightly changed the trainer function for extending the vocab) with the vocab_size of 5000 for this model.

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
    "warmup_steps" : 3000,
    "vocab_size : 5000
}
```

**Datasets:** Phase1: Harrypotter & LOTR complete series.
              Phase2: Dune complete series.

**Training details:**

1. The model was trained on the dataset of >4 million raw text tokens, with the block_size of 256 tokens and embedding dimensions of 256.
2. Used RoPE to extend the context length during inference and enhancing token relations through relative position embeddings during training.
3. Used GELU activation function with tanh approximation in the Feed forward layer as it works better(But the exact reason is still not clear).
4. Implemented Grouped Multi Query attention for faster training. But I didn't see much difference in the training time as my model is too small.
5. I've chosen RMS Normalization because it simplifies the process by removing extra mean calculations, which has no impact, as mentioned in the paper.
6. Normalized both the Inputs and Outputs of the model for better numerical stability and to avoid saddle points during the gradient optimization.
7. I used the AdamW optimizer with a weight decay of 0.99. Although a higher value slows down model convergence, Still I used it for minimizing the overfitting as much as possible.
8. I trained this model in two phases with two different learning schedulers. In the first phase, I used the CyclicLR scheduler, halving the amplitude scaling for each cycle, and in the second phase, I used the  Cosine scheduler with linear warmup. Although AdamW ideally pairs with the linear warmup cosine scheduler, I chose CyclicLR because it effectively explores the parameter space and improves the training convergence quickly.
9. I trained this model using full precision floating point numbers because bfloat16 made the training slower for some reason. I also tried mixed precision training with PyTorch's AMP package, but it caused the numercial instability in the gradients.
10. At first, the gradient and parameters ratio was high while training. Later on, it went down gradually and got closer to 1.

The overall training process went smoothly, and the final model tried to effectively captures data patterns, sentence structures and some coherence in the sentences despite its small size. However, the model could be enhanced by adding more data and parameters to generate even more coherent sentences. Luckily the model didn't overfit much, indicating there are no bugs in the model architecture and training code. This can be scalable for a large language model with more parameters and with minor modifications to the architecture.

While the model is able to mimic the authors style of writing by generating conversations with back-and-forth dialogue between the characters and grasping the story, but it lacks the overall clarity in the generated text, However it did achieves some coherence in sentences. Additionally, the model is capable of generating well-structured sentences with proper punctuation to a great extent.



**Next Steps:**
1. Scaling the model to 500M parameters and training it on a huge dataset using 1.58-bit quantization.
