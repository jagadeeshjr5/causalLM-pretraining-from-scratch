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

param_dict = {
    "batch_size" : 56,
    "block_size" : 256,
    "embd" : 256,
    "n_heads" : 8,
    "n_kvheads" : 4,
    "head_size" : 256 // 8,
    "dropout" : 0.2,
    "n_layer" : 3,
    "learning_rate" : 1.5e-4,  #3e-4
    "n_splits" : 4,
    "norm_eps" : 2e-4,
    "device" : 'cuda',
    "accumulation_steps" : 1,
    "epochs" : 1,
    "eval_iters" : 150,
    "start_pos" : 0,
    "temperature" : 0.5,
    "warmup_steps" : 1000
}



##Rotary positional Embeddings
def precompute_theta_pos_frequencies(head_size: int, block_size: int, device: str, theta: float = 1e5):

    assert head_size % 2 == 0, "Dimension must be divisible by 2"

    theta_numerator = torch.arange(0, head_size, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_size)).to(device) # (Dim / 2)

    m = torch.arange(block_size, device=device)

    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):

    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    x_rotated = x_complex * freqs_complex

    x_out = torch.view_as_real(x_rotated)
    
    x_out = x_out.reshape(*x.shape)
    
    return x_out.type_as(x).to(device)


##Helper function for Grouped multi Query Attention
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, block_size, n_kvheads, head_size = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(batch_size, block_size, n_kvheads, n_rep, head_size)
        .reshape(batch_size, block_size, n_kvheads * n_rep, head_size)
    )


##Grouped multi query attention(With ROPE)
class Attention(nn.Module):
    def __init__(self, embd: int, n_heads: int, dropout: float, n_kvheads: int):
        super().__init__()

        self.embd = embd
        self.n_heads = n_heads
        self.dropout = dropout
        self.n_kvheads = n_kvheads
        self.n_rep = self.n_heads // self.n_kvheads
        self.head_size = self.embd // self.n_heads

        assert self.embd % self.n_heads == 0

        self.qlin = nn.Linear(self.embd, self.n_heads * self.head_size, bias=False)
        self.klin = nn.Linear(self.embd, self.n_kvheads * self.head_size, bias=False)
        self.vlin = nn.Linear(self.embd, self.n_kvheads * self.head_size, bias=False)
        
        nn.init.kaiming_normal_(self.qlin.weight, mode='fan_in')
        
        nn.init.kaiming_normal_(self.klin.weight, mode='fan_in')
        
        nn.init.kaiming_normal_(self.vlin.weight, mode='fan_in')
        
        self.outproj = nn.Linear(self.embd, self.embd, bias=None)
        
        nn.init.kaiming_normal_(self.outproj.weight, mode='fan_in')

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.tril(torch.ones(1, 1, 2048, 2048)))  # Assuming max block_size is 2048

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, block_size, embd = x.shape

        wq = self.qlin(x)
        wk = self.klin(x)
        wv = self.vlin(x)

        wq = wq.view(batch_size, block_size, self.n_heads, self.head_size)
        wk = wk.view(batch_size, block_size, self.n_kvheads, self.head_size)
        wv = wv.view(batch_size, block_size, self.n_kvheads, self.head_size)
        
        wq = apply_rotary_embeddings(wq, freqs_complex, device=x.device)
        wk = apply_rotary_embeddings(wk, freqs_complex, device=x.device)

        keys = wk.repeat(1, 1, self.n_rep, 1)
        values = wv.repeat(1, 1, self.n_rep, 1)

        wq = wq.transpose(1, 2).to(x.device)
        keys = keys.transpose(1, 2).to(x.device)
        values = values.transpose(1, 2).to(x.device)

        attn_wei = torch.matmul(wq, keys.transpose(-2, -1)) * (1.0 / keys.size(-1) ** 0.5)
        attn_wei.masked_fill_(self.mask[:, :, :block_size, :block_size] == 0, float('-inf'))
        attn_wei = F.softmax(attn_wei, dim=-1)
        attn_wei = self.attn_dropout(attn_wei)
        attn_wei = torch.matmul(attn_wei, values)

        out = attn_wei.transpose(1, 2).contiguous().view(batch_size, block_size, embd)

        return self.resid_dropout(self.outproj(out))
    

    
    
    ##RMS Norm(Non parameterized)
    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def _norm(self, x: torch.Tensor):

            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        def forward(self, x: torch.Tensor):
            #return self.weight * self._norm(x.float()).type_as(x)
            return self._norm(x.float()).type_as(x)
        
    

    ##Feed Forward Layer with GELU activation, Tanh approximation
    class FeedForwardLayer(nn.Module):
        def __init__(self):
            super(FeedForwardLayer, self).__init__()
            
            self.w1 = nn.Linear(param_dict['embd'], 4 * param_dict['embd'], bias=True)
            self.w2 = nn.Linear(4 * param_dict['embd'], param_dict['embd'], bias=True)
            self.w3 = nn.Linear(param_dict['embd'], 4 * param_dict['embd'], bias=True)
            
            nn.init.kaiming_normal_(self.w1.weight, mode='fan_in')
            nn.init.constant_(self.w1.bias, 0)
            
            nn.init.kaiming_normal_(self.w2.weight, mode='fan_in')
            nn.init.constant_(self.w2.bias, 0)
            
            nn.init.kaiming_normal_(self.w3.weight, mode='fan_out')
            nn.init.constant_(self.w3.bias, 0)
            
        def forward(self, x):
            
            gelu = F.gelu(self.w1(x), approximate='tanh')
            x_V = self.w3(x)
            x = gelu * x_V
            x = self.w2(x)
            return x
        


##Transformer block
class TranformerBlock(nn.Module):
    def __init__(self, n_embd : int, n_head : int):
        super().__init__()
        self.swsa = Attention(param_dict['embd'], param_dict['n_heads'], param_dict['dropout'], param_dict['n_kvheads'])
        self.ffwd = FeedForwardLayer()
        self.rms1 = RMSNorm(param_dict['embd'], eps=param_dict['norm_eps'])
        self.rms2 = RMSNorm(param_dict['embd'], eps=param_dict['norm_eps'])
        self.rms3 = RMSNorm(param_dict['embd'], eps=param_dict['norm_eps'])
        self.rms4 = RMSNorm(param_dict['embd'], eps=param_dict['norm_eps'])

    def forward(self, x : torch.Tensor, start_pos : int, freqs_complex : torch.Tensor):
        x = x + self.rms3(self.swsa(self.rms1(x), start_pos, freqs_complex))
        x = x + self.rms4(self.ffwd(self.rms2(x)))
        return x
    

##Helper function for calculating Gradient and Parameter norms
def calculate_norm(params, param_type='GRADIENTS'):
    norms = []
    for param in params:
        if param_type == 'GRADIENTS' and param.grad is not None:
            norm = torch.norm(param.grad)
            norms.append(norm.item())
        elif param_type == 'PARAMETERS':
            norm = torch.norm(param)
            norms.append(norm.item())
    if len(norms) == 0:
        return 0.0  # Return 0 if no gradients or parameters found
    else:
        mean_norm = sum(norms) / len(norms)
        return mean_norm
    


##Final Model Class
class LanguageModel(nn.Module):

    def __init__(self):
        super(LanguageModel, self).__init__()
        self.token_embedding_table = nn.Embedding(param_dict['vocab_size'], param_dict['embd'])
        self.blocks = nn.Sequential(*[TranformerBlock(param_dict['embd'], n_head=param_dict['n_heads']) for _ in range(param_dict['n_layer'])])
        self.rn_f = RMSNorm(param_dict['embd'],  eps=param_dict['norm_eps']) # final layer norm
        self.lm_head = nn.Linear(param_dict['embd'], param_dict['vocab_size'])
        
        self.freqs_complex = precompute_theta_pos_frequencies(param_dict['head_size'], param_dict['block_size'] * 2, device=param_dict['device'])
        
        nn.init.kaiming_normal_(self.lm_head.weight, mode='fan_out')
        torch.nn.init.normal_(self.token_embedding_table.weight, mean=0.0, std=0.5)


    def forward(self, x, targets=None):
        batch_size, block_size = x.shape
        
        start_pos = param_dict['start_pos']
        
        freqs_complex = self.freqs_complex[start_pos:start_pos + block_size]

        tok_emb = self.token_embedding_table(x)

        x = tok_emb
        for block in self.blocks:
            x = block(x, start_pos, freqs_complex=freqs_complex)
        x = self.rn_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            batch_size, block_size, embd = logits.shape
            logits = logits.view(batch_size*block_size, embd)
            targets = targets.view(batch_size*block_size)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def num_parameters(self):
        
        n_params = str(sum(p.numel() for p in self.parameters())/1e6) + ' M parameters'
        
        return n_params
    
    torch.autograd.set_detect_anomaly(True)
    
    def start_train(self, data, optimizer, checkpoint_path=None):
        
        wandb.init(project="JARV1")
        wandb.config = {"learning_rate": param_dict['learning_rate'], "epochs": param_dict['epochs'], "batch_size": param_dict['batch_size'], "embd" : param_dict['embd']}
        
        self.to(param_dict['device'])
        
        wandb.watch(self, log="all", log_freq=100)
        
        total_steps = param_dict['epochs'] * len(dataloader)
        
        #warmup_lr = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=param_dict['warmup_steps'], num_training_steps=total_steps)
        warmup_lr = warmup.LinearWarmup(optimizer, warmup_period=param_dict['warmup_steps'])
        cosine_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1.5e-5)
        #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = 5e-5, max_lr=param_dict['learning_rate'], step_size_up=2500, step_size_down=15000, cycle_momentum=False, mode='triangular2')
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            cosine_lr.load_state_dict(checkpoint['scheduler_state_dict'])
        
        
        batches = len(data)
        
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            
        #scaler = GradScaler()
        
        outputs = []
        inputs = []
        
        table = wandb.Table(columns = ["Input", "Output"])
        
                
        for epoch in tqdm(range(param_dict['epochs'])):
            self.train()
            optimizer.zero_grad()

            for batch, (x_batch, y_batch) in tqdm(enumerate(data), total=batches):

                #if checkpoint_path:
                #    if epoch <= checkpoint.get('epoch', 0) and batch <= checkpoint.get('batch', 0):
                #        if epoch == checkpoint.get('epoch', 0) and batch == checkpoint.get('batch', 0):
                #            print(f'Reinstating training from {epoch}th epoch, {batch + 1}th batch...')
                #        continue

                
                
                x, y = x_batch.to(param_dict['device']), y_batch.to(param_dict['device'])

                logits, loss = self(x, y)
                optimizer.zero_grad(set_to_none=True)
                #scaler.scale(loss).backward()
                loss.backward()
                
                #scaler.unscale_(optimizer)

                #torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0, norm_type=1.0)

                if (batch + 1) % param_dict['accumulation_steps'] == 0:
                    #scaler.step(optimizer)
                    optimizer.step()
                    #scaler.update()
                    
                    with warmup_lr.dampening():
                        if warmup_lr.last_step + 1 >= param_dict['warmup_steps']:
                            cosine_lr.step()
                    if warmup_lr.last_step + 1 >= total_steps:
                        break

                    #cosine_lr.step()
                    wandb.log({"batch": batch, "warmup_lr": optimizer.param_groups[0]['lr']})


                if batch % 500 == 0:
                    losses, perplexity = estimate_loss()
                    #print(f"               step {batch} / {len(dataloader)}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                    wandb.log({"batch": batch, "loss": losses, "perplexity" : perplexity})


                    mean_gradient_norm = calculate_norm(self.parameters(), param_type='GRADIENTS')
                    mean_parameter_norm = calculate_norm(self.parameters(), param_type='PARAMETERS')

                    norm = mean_parameter_norm / mean_gradient_norm

                    wandb.log({"batch": batch, "norm": norm, "gradient_norm" : mean_gradient_norm, "parameter_norm" : mean_parameter_norm})
                    print(f"Ratio of mean gradient norm to mean parameter norm: {norm}, gradient norm: {mean_gradient_norm}, parameter norm: {mean_parameter_norm}")

                optimizer.zero_grad()

                if batch in list(range(5000, batches + 5000, 5000)) or batch == batches-1:
                    checkpoint = {
                        'epoch': epoch,
                        'batch': batch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': cosine_lr.state_dict()
                        #'dataloader_state_dict': random_seed
                    }
                    torch.save(checkpoint, r'C:\Transformers Pretraining\checkpoints\checkpoint3c1.pth')
                    print(f"Saved checkpoints at epoch: {epoch} and batch: {batch}.")
                    output = self.inference(1000, temperature=0.75, top_k=5, top_p=0.92)

                    table.add_data("Jagadeesh is the king of", output)


            losses, perplexity = estimate_loss()
            wandb.log({"Epoch": epoch, "loss": losses, "perplexity" : perplexity})

            wandb.log({"Table Name": table})


        return model
        
    def save(self, path):
        
        torch.save(self.state_dict(), f'{path}/model.pth')
        
        return f'Successfully saved model to {path}.' 
        
    def load(self, checkpoint_path):
        
        checkpoint = torch.load('C:\Transformers Pretraining\checkpoints\checkpoint3c2epoch3v3.pth')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.to('cuda')

        #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to('cuda')
        
        return self
        
    @torch.no_grad
    def inference(self, max_new_tokens, temperature, top_k=None, top_p=None):
        self.eval()
        #prompt = str(input("Prompt here: "))
        prompt = "Harry talked to Gandalf about"
        data = torch.tensor(tok.encode(prompt), dtype=torch.long)
        context = torch.tensor(data.view(1, len(data)), device=param_dict['device'])
        block_size = param_dict['block_size']

        for _ in range(max_new_tokens):
            idx_cond = context[:, -block_size:]
            logits, loss = self(idx_cond)

            if temperature:
                logits = logits[:, -1, :] / temperature
            else:
                logits = logits[:, -1, :]

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                sorted_logits[sorted_indices_to_remove] = -float('Inf')

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, idx_next), dim=1)

        return "".join(tok.decode(context[0].tolist()))


