#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 00:42:18 2024

@author: lurker18
"""

from typing import Dict

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPImageProcessor,
    LlamaTokenizer,
    )

def get_tokenizer(language_model_name: str) -> "Tokenizer":
    if "opt" in language_model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            language_model_name, padding_size = "right", use_fast = False
        )
        return tokenizer
    
    elif "Llama" in language_model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            language_model_name, padding_side = "right", use_fast = False
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    else:
        raise NotImplementedError(
            f"Tokenizer for language_model_name: {language_model_name} is not implemented."
        )
        

def get_processor(model_config: Dict) -> "Processor":
    language_model_name = model_config["language_model_name"]
    model_type = model_config["model_type"]
    
    if "git" in model_type:
        processor = AutoProcessor.from_pretrained("microsoft/git-base")
        processor.image_processor = CLIPImageProcessor.from_pretrained(
            model_config["vision_model_name"]
        )
        
    else:
        raise NotImplementedError(f"Processor for model_type: {model_type} is not implemented.")
        
    processor.tokenizer = get_tokenizer(language_model_name)
    
    return processor