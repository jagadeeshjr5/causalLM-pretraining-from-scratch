import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim.lr_scheduler as lr_scheduler
from minbpe import minbpe as bpe
from typing import List
import wandb
from tqdm.auto import tqdm


class CustomDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, index):
        return self.data[index:index+self.block_size], self.data[index+1:index+self.block_size+1]
    

    
#Andrej karpathy's minbpe tokenizer

#!wget https://github.com/karpathy/minbpe.git

class Tokenizer:
    def __init__(self):
        self.tokenizer = bpe.RegexTokenizer()
    def train(self, text : str, vocab_size : int):
        self.tokenizer.train(text, vocab_size)
        return None
    def encode(self, text : str):
        return self.tokenizer.encode(text)
    def decode(self, enc_list : List):
        return self.tokenizer.decode(enc_list)
    def vocab(self):
        return self.tokenizer.vocab
    def save(self, path):
        self.tokenizer.save(path)
        return f'Tokenizer saved to {path}'
    def load(self, path):
        self.tokenizer.load(path)
        return f'Successfully loaded tokenizer'
    
    
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        
        data = train_data if split == 'train' else val_data
        
        dataset = CustomDataset(data, block_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        losses = torch.zeros(eval_iters)
        for batch, (x_batch, y_batch) in enumerate(dataloader):
            if batch == eval_iters:
                break            
            x, y = x_batch.to(device), y_batch.to(device)
            logits, loss = model(x, y)
            losses[batch] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    

filenames = [
    'crime-and-punishment-fedor-mikhailovitch-distoievski.txt',
    '2. Oliver Twist Author Charles Dickens-compressed.txt',
    '3. David Copperfield Author Charles Dickens-compressed.txt',
    'don-quixote-miguel-de-cervantes.txt',
    '3. The Lost World Author Arthur Conan Doyle.txt',
    '2. Twenty Thousand Leagues Under the Seas Author Jules Verne.txt',
    '1. The Time Machine Author H. G. Wells.txt'
]

text = ''
for filename in filenames:
    with open(f'C:/Transformers Pretraining/JARV1-MicroLM/data/{filename}', 'r', encoding='latin1') as f:
        text += f.read()
        

tok = Tokenizer()

tok.load(r'C:\Transformers Pretraining\New folder\tok3k.model')

data = torch.tensor(tok.encode(text), dtype=torch.long)

vocab_size = len(tok.vocab())

n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

dataset = CustomDataset(train_data, block_size)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

wandb.init(project="JARV1")

wandb.config = {"learning_rate": learning_rate, "epochs": 1, "batch_size": batch_size, "embd" : embd}

model = GPTLanguageModel()

device = 'cuda'
model = GPTLanguageModel()
model = model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[12000], gamma=0.1)
        
wandb.watch(model, log="all", log_freq=10)

accumulation_steps = 4
for epoch in tqdm(range(1)):
    model.train()
    optimizer.zero_grad()
    
    for batch, (x_batch, y_batch) in tqdm(enumerate(dataloader)):
        x, y = x_batch.to(device), y_batch.to(device)

        if batch % 100 == 0 or batch == 1000 - 1:
                losses = estimate_loss()
                #print(f"               step {batch} / {len(dataloader)}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
                wandb.log({"batch": batch, "loss": losses})

        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        if (batch + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
            scheduler.step()
        
    losses = estimate_loss()
    print(f"Epoch {epoch+1}/1: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    wandb.log({"Epoch": epoch, "loss": losses})
    
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(tok.decode(model.generate(context, max_new_tokens=1000)[0].tolist()))