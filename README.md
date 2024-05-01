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

**Generated text:**

```
He will be here soon to find a new home on Arrooen, and we'll
be at least four hundred years ago. The Guild Bank has its uses.
A: "The Gammu."
   A deep grin came up on the city and stared at him in the face of
the man's voice. "The Harkonnens were saying," he said. "We
will not want that, I know." He turned, his face lighter with a
fashioned armor. "We are not to be able to bind them or do this,
possibility."
   She looked down the table at the Landsraad. The Guildsman's
voice came out flat and a smile on his face.
   "The Bene Gesserit have no idea how the Sardaukar are.
   "The Sardaukar will not be a Baron," she said.
   He turned his face against the light, saw the door outlined
cushion in her eyes. The old Fremen remained in the room. She
was not there. It was a man of many places, but a man with a hood
of hair was a black and a stilltent. She had a pale, dark eyes. He
was still out in her mind, his eyes wide. His eyes were wide and his
features, and she knew it could not be. His eyes were the eyes, its
eyes fixed on her face.
   "I have only the right hand of the Emperor in the Emperor," she
said, his gaze left and right. "Alia, I'm not your father's son."
   She stared down behind her. "You have not been a child."
   "Yes, m'Lord. You and Hayt."
   "And you're the one the young woman?"
   "A man of many Fremen?"
   "I am one who has not seen a Mentat, my Lady. I am the mate," she
said. "But my inherited and I have the Atreides banner."
   "I have not heard about him."
   "Then you must know he was . . . I must have him alone and
wait."
   "I have." He nodded to himself. "Yes. You know, Paul."
   "But you must be dangerous to me."
   "You're sure he's dead," he said.
   "He's a Mentat. He is so close to the Sardauko. I know you're
conditioning. You're the most dangerous, a genesis of the Attendable
Muad'Dib!"
   "He has been here, Muad'Dib, what does he want of me?"
   "It was . . . a Fremen?"
   "Yes, I have not thought of myself as I did. I have only one
children of the malignant, a male."
   "And you said you could be the Atreides," she whispered. "The Empire
are not a Fremen. The Fremen know melange is noted for the
Harkonnens."
   "I am the one who killed the Mentat," she said. "The Sardaukar
must know what was in the desert. The Fremen have a new Bene
Gesserit."
   "But I am not a Tleilaxu."
   She looked away from her, feeling the painfulness of his
mouth. She felt a pang for the first time in the old days before the
Sardaukar were in her life. The thing could be a ghola. He had seen
this by being ascurious as the Tyrant. And here is what I have done.
   He spoke of his own cart and he thought: The Tleilaxu had been
always taken from a Reverend Mother and the Tleilaxu. The
Sardaukar could be seen on this. She could not make her own oï¬spring,
but Schwangyu had been infant, but he was a mask. She was not
suspensor in a way of escape. He knew what he had been and
had gone to this planet, the gholas: Mother Superior. And
Honored Matres had taught him that Teg was always thinking
about the Bene Gesserit.
   Teg was right.
   "The Tleilaxu Masters have some of the Bashar's men," she said. And as
the gholas were only three times the Bashar had been ascribed to the
Sisterhood, that many times-the Tleilaxu, of course. He had been a
predator. It had been the first to bring in with the T-probe and now
returned. The Bashar's favorite ghola. It had come out of the Scatter
beside me and . . . and that was not the only person to reject a mentat. Schwangy-
was-the most important thing of the ghola, a Bene Gesserit, an
interesting person in a Reverend Mother with a Reverend Mother
who had been on her way to the Sisterhood, her life in a mentat
replacemor to the Bene Gesserit of his Mentats and not by all in a
recalling of the Honored Matres, the Bene Gesserit, had been
tempted when their ghola-atomics of Tarota, and the Sisterhood
had never suspected him. It was an extensive thing which had once
became into the ghola. She had known what had happened in this
dembrane in the ghola. The Tleilaxu had been costingly
deluable on the Spacing Guild, a Mother Sympocaused on the
Honored Matres. But what of the ghola?
```

**Next Steps:**
1. Scaling the model to 500M parameters and training it on a huge dataset using 1.58-bit quantization.
