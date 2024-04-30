import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torch.optim.lr_scheduler as lr_scheduler
from minbpe import minbpe as bpe
import wandb
from typing import List
from tqdm.auto import tqdm
import torch
from torch.cuda.amp import autocast, GradScaler
import inspect
import pytorch_warmup as warmup
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_cosine_schedule_with_warmup

import warnings
warnings.filterwarnings('ignore')

from model import LanguageModel, param_dict

class CustomDataset(Dataset):
    def __init__(self, data, block_size : int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, index):
        return self.data[index:index+self.block_size], self.data[index+1:index+self.block_size+1]
    

    
#Andrej karpathy's minbpe tokenizer with a slight change in the trainer function to extend the tokenizer vocabulary.

#!wget https://github.com/karpathy/minbpe.git

class Tokenizer:
    def __init__(self):
        self.tokenizer = bpe.RegexTokenizer()
    def train(self, text : str, vocab_size : int):
        self.tokenizer.train(text, vocab_size)
        return None
    def encode(self, text : str):
        return self.tokenizer.encode(text)
    def decode(self, enc_list):
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
    perplexity_out = {}
    model.eval()
    
    for split in ['train', 'val']:
        
        data = train_data if split == 'train' else val_data
        
        dataset = CustomDataset(data, param_dict['block_size'])
        dataloader = DataLoader(dataset, batch_size=param_dict['batch_size'], shuffle=True)
        
        losses = torch.zeros(param_dict['eval_iters'])
        perplexity_tensor = torch.zeros(param_dict['eval_iters'])
        for batch, (x_batch, y_batch) in enumerate(dataloader):
            if batch == param_dict['eval_iters']:
                break            
            x, y = x_batch.to(param_dict['device']), y_batch.to(param_dict['device'])
            logits, loss = model(x, y)
            perplexity = torch.exp(loss)
            losses[batch] = loss.item()
            perplexity_tensor[batch] = perplexity.item()
        out[split] = losses.mean()
        perplexity_out[split] = perplexity_tensor.mean()
        
    model.train()
    return out, perplexity_out
    

filenames = [
        'Harry-Potter-the-Complete-Series.txt',
    'JRR-Tolkien-Lord-of-the-Rings-Collection.txt',
    'Frank-Herberts-Dune-Saga-Collection-Books-1-6-by-Frank-Herbert.txt',
    'Star-Wars-The-Old-Republic-Revan-PDF-Room.txt'
]

text = ''
for filename in filenames:
    with open(f'data/{filename}', 'r', encoding='latin1') as f:
        text += f.read()
        

tok = Tokenizer()

tok.load(r'tokenizer\tok5kV2.model')

param_dict['vocab_size'] = len(tok.vocab())

data = torch.tensor(tok.encode(text), dtype=torch.long)

n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

dataset = CustomDataset(train_data, param_dict['block_size'])

torch.manual_seed(4235246) #4235245
np.random.seed(4235246)

random_sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(4235246))

dataloader = DataLoader(dataset, batch_size=param_dict['batch_size'], sampler=random_sampler)

model = LanguageModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=param_dict['learning_rate'], weight_decay=0.99, betas=(0.9,  0.95))

model.load(optimizer=optimizer, checkpoint_path='checkpoints\checkpoint3c2-pretraining.pth')

model.start_train(data=dataloader, optimizer=optimizer, train_data=train_data, val_data=val_data, checkpoint_path='checkpoints\checkpoint3c2-pretraining.pth')

