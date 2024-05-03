# causalLM-pretraining-from-scratch

**This is a 5M parameter Causal Language model based on Transformer Decoder only architecture trained on just >4M raw text tokens on an RTX 3060 gaming GPU.**

**Arcitecture details:**
1. Transformer - Decoder Only Arcitecture.
2. Context length - 256 & embd dims - 256.
3. Grouped Multi Query attention with 2:1 query, key ratio.
4. ROPE with theta of 1e5.
5. GELU activation with tanh approximation.
6. RMS Norm(de-parameterised) applied for both inputs and outputs.
7. Remaining vanilla transformer stuff.

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
2. Used RoPE to enhance the token relations through relative position embeddings during training.
3. Used GELU activation function with tanh approximation in the Feed forward layer as it works better(But the exact reason is still not clear).
4. Implemented Grouped Multi Query attention for faster training. But I didn't see much difference in the training time as my model is too small.
5. I've chosen RMS Normalization because it simplifies the process by removing extra mean calculations, which has no impact, as mentioned in the paper.
6. I've de-parameterized the normalization layers for faster training. Moreover, it didn't affect the model's learning curve and accuracy much(Need to do further experimentation on this).
7. Normalized both the Inputs and Outputs of the model for better numerical stability and to avoid saddle points during the gradient optimization.
8. I used the AdamW optimizer with a weight decay of 0.99. Although a higher value slows down model convergence, Still I used it for minimizing the overfitting as much as possible.
9. I trained this model in two phases with two different learning schedulers. In the first phase, I used the CyclicLR scheduler, halving the amplitude scaling for each cycle, and in the second phase, I used the  Cosine scheduler with linear warmup. Although AdamW ideally pairs with the linear warmup cosine scheduler, I chose CyclicLR because it effectively explores the parameter space and improves the training convergence quickly.
10. I trained this model using full precision floating point numbers because bfloat16 made the training slower for some reason. I also tried mixed precision training with PyTorch's AMP package, but it caused the numercial instability in the gradients.
11. At first, the gradient and parameters ratio was high while training. Later on, it went down gradually and got closer to 1.

The overall training process went smoothly, and the final model tried to effectively captures data patterns, sentence structures and some coherence in the sentences despite its small size. However, the model could be enhanced by adding more data and parameters to generate even more coherent sentences. Luckily the model didn't overfit much, indicating there are no bugs in the model architecture and training code. This can be scalable for a large language model with more parameters and with minor modifications to the architecture.

While the generated output is not that great but it is able to mimic the authors style of writing by generating conversations with back-and-forth dialogue between the characters and grasping the story. But it lacks the overall clarity in the generated text, However it did achieves some coherence in sentences. Additionally, the model is capable of generating well-structured sentences with proper punctuation to a great extent.

**Loss curve:**

In the loss curve below, there is a clear decreasing trend in both the training and validation losses, suggesting that the model is not overfitting. However, there are a few minor ridges; these are because of the cyclicLR scheduler.

![Alt text](https://github.com/jagadeeshjr5/causalLM-pretraining-from-scratch/blob/6cdd9acf2881bd9c9c0e0d3dca407b66d2cf6fa9/phase1_loss_curve.png)

**Generated sample:**

```
Leto had not come back from the first, Sheeana's party. It had not
realized her first hours had passed but the memory-lying of her
response, which Taraza was requests, she thought. The Tleilaxu
was being torn by that infinite universe, no longer human, but they
had not seen her at this moment.
   Sheeana was a Reverend Mother and her trades were seated
with Fish Speaker training, but a cynical and unreal. The Land of
the Bene Face Dancers was a Reverend Mother. The Suk, too, was
not for this: he knew her psychologists.
    "We must have no time to use the Empire!" Sheeana said.
She had been the last words of the Bene Gesserit. The Guildsman was a
multitudinal and capable of dispute with a diversion of
destructive data. They had to be more than one male for her in the
family provoked the Breeding Probe, the Ixians had its most influence. And the
Mother Superior, the Guild naturally truly existent the Tyrans, and the
power to which they were capably profound. She knew a Mentat could
become another one, but that was more than two days or so.
    The Bene Gesserit were inevitly the way Idaho had told her the way
she'd been out there on the Bene Gesserit. The Bene Gesserit had
sensed his intimate knowledge that the Bene Gesserit would not
be denied their original memories.
   "We do not understand this," Odrade said.
   Teg glanced around the room. He could only see a face that
werea man of influence on him.
   The prize of her face flowed over the desert. He knew, of how
that would be an organization and not association with that
danger, is the future of the Bene Tleilax and the Bashar.
    She saw the Bashar in his awareness: The universe of the Bene
Gesserit, but the Bashar was one of the Tyrant's old patterns, and
the Honored Matrior.
   Those who had done what she had learned from her as "Power.
Many would know."
  Our genes. The Bashars were so good."
                   The Mother Superior was not at all times but not right
   that the Miles Tyrant himself had learned to read about the
  Mentat Bene Gesserit. The Bene Tleilax as well as the Sisterhood.
```

**Next Steps:**
1. Scaling the model to 500M parameters and training it on a huge dataset using 1.58-bit quantization.
