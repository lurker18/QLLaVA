#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:35:25 2024

@author: lurker18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # Number of the quaries
    n_kv_heads: Optional[int] = None # Number of heads for the k and v
    vocab_size: int = -1 # This will be set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    
    # Needed for KV Cache
    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    device: str = None
    
class Transformer(nn.Module):
    def __init__ (self, args: ModelArgs) -> None:
        super().__init__()
        
        assert args.vocab_size != -1, "vocab size must be set"
        
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlocks(args))
        
        self.norm = RMSNorm(args.dim, eps = args.norm_eps)
        