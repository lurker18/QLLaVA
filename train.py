#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 18:06:54 2024

@author: lurker18
"""

import os
from typing import Any

import fire
import deepspeed
import torch
import yaml

from transformers import TrainingArguments

from Code.datasets.utils import get_dataset
from Code.Models.utils import (apply_lora_model, load_model, load_pretrained_weight, set_trainable_params, unload_and_merge_lora)
from Code.Models.vision_language_trainer import VisionLanguageTrainer as Trainer

GitLLMForCausalLM = Any

def main(config_file: str, local_rank: int = 0):
    with open(config_file, "r") as i_:
        config = yaml.safe_load(i_)
        model_config = config["model_config"]
        training_config = config["training_config"]
        
    if os.environ.get("WANDB_NAME") is not None:
        training_config["output_dir"] = os.path.join(
            training_config["output_dir"], os.environ["WANDB_NAME"]
        )
        
    # distributed learning
    deepspeed.init_distributed()
    
    # config to what to freeze and finetune
    keys_to_finetune = config["model_config"]["keys_to_finetune"]
    keys_to_freeze = config["model_config"]["keys_to_freeze"]
    
    # Dataset load
    train_dataset, val_dataset = get_dataset(config)
    
    training_args = TrainingArguments(**training_config)
    
    
    # load model
    model = load_model(model_config)
    
    if model_config["use_lora"]:
        model = apply_lora_model(model, model_config)
        
    # Load pretrained weight
    if model_config.get("pretrained_path") is not None:
        print("load pretrained")
        load_pretrained_weight(model, model_config["pretrained_path"])
        print(f'Successfully loading pretrained weights from {model_config["pretrained_path"]}')
        
    # Set trainable params
    trainable_list, untrainable_list = set_trainable_params(
        model, keys_to_finetune, keys_to_freeze, train_lora = model_config["use_lora"]
    )
    print("trainable_list", trainable_list)
    print("untrainable_list", untrainable_list)
    
    trainer = Trainer(
        model = model,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        args = training_args,
    )
    
    with torch.autocast("cuda"):
        trainer.train()
        
    # Save the final checkpooint
    if os.environ.get("WANDB_NAME") is not None:
        final_save_path = os.path.join(
            training_config["output_dir"], os.environ["WANDB_NAME"] + "_final"
        )
    else:
        final_save_path = os.path.join(training_config["output_dir"], "final_model")
        
    
    if model_config["use_lora"]:
        model = unload_and_merge_lora(model, model_config)
    model.save_pretrained(final_save_path)
    train_dataset.datasets[0].processor.save_pretrained(final_save_path)
    

if __name__ == "__main__":
    fire.Fire(main)
    
