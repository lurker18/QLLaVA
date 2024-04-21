# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 01:07:49 2024

@author: Nova18
"""


import torch
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor

from llama2 import ModelArgs, Transformer
from transformers import LlamaForCausalLM, LlamaTokenizer

class Llama:
    def __init__(self, model: Transformer, tokenizer: LlamaTokenizer, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args
        
    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, max_seq_len: int, max_batch_size: int, device: str):
        
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
            
        model_args: ModelArgs = ModelArgs(
            max_seq_len = max_seq_len,
            max_batch_size = max_batch_size,
            device = device,
            **params
        )
        
        if device == "cuda":
            torch.set_default_dtype(torch.float16)
        else:
            torch.set_default_dtype(torch.bfloat16)
        
        tokenizer = LlamaTokenizer.from_pretrained(Path(tokenizer_path))
        model_args.vocab_size = tokenizer.vocab_size
        
        model = LlamaForCausalLM.from_pretrained(Path(checkpoints_dir))
        model.to(device)
        
        return Llama(model, tokenizer, model_args)

if __name__ == "__main__":
    torch.manual_seed(0)
    
    allow_cuda = True
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"
    
    model = Llama.build(
        checkpoints_dir = "llama-2-13b/",
        tokenizer_path = "llama-2-13b/tokenizer.model",
        max_seq_len = 1024,
        max_batch_size = 1,
        device = device
    )
    
    print("All Finished!!!")



