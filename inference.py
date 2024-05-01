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
    
tok = Tokenizer()

tok.load(r'tokenizer\tok5kV2.model')

param_dict['vocab_size'] = len(tok.vocab())

model = LanguageModel()
model.load(optimizer=None, checkpoint_path='checkpoints\checkpoint3c2-pretraining.pth')
model.eval()

print(model.inference( max_new_tokens=1000, temperature=1.25, top_k=5, top_p=0.92))