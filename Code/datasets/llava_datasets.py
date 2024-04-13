#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 01:05:23 2024

@author: lurker18
"""

import os
from typing import Dict

import cv2
import numpy as np
import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset as HFDataset
from PIL import Image

from Code.datasets.base_datasets import IGNORE_INDEX, BaseDataset

HFProcessor = "HFProcessor"


class LlavaDataset(BaseDataset):
    """Dataset for LLaVA"""
    def __init__(self,
                 loaded_dataset: HFDataset,
                 processor:HFProcessor,
                 max_length: int,
                 is_inference: bool,
                 dataset_root: str,
    ):
        super(LlavaDataset, self).__init__(is_inference)
        self.loaded_dataset = loaded_dataset
        self.max_length = max_length
        self.processor = processor
        self.is_inference = is_inference
        self.dataset_root = dataset_root
        
    @classmethod
    def create(
            cls,
            dataset_config: Dict,
            processor: HFProcessor,
            max_length: int,
            split: str = "train",
            is_inference: bool = False,
    ):
        """
        Args:
            dataset_config: dataset configuration
            processor: HuggingFace Processor class
            split: data split. "train" or "val"
            is_inference: inference mode or not
        """
        jsonl_path = dataset_config.get("jsonl_path", None)
        n_train = dataset_config["n_train"]
        n_val = dataset_config["n_val"]
        if jsonl_path is not None:
            import json
            
            with open(jsonl_path) as f:
                jsonl_datasets = json.load(f)
            split_datasets = {
                "train": jsonl_datasets[:n_train],
                "test" : jsonl_datasets[n_train : n_train + n_val],
            }
        else:
            print("Insert the correct dataset path!!")
        
        
        if split == "train":
            return cls(
                split_datasets["train"],
                processor,
                max_length,
                is_inference,
                dataset_config["dataset_root"],
            )
        
        elif split == "validation":
            return cls(
                split_datasets["test"],
                processor, 
                max_length,
                is_inference,
                dataset_config["dataset_root"],
            )
        else:
            raise ValueError("given split is invalid")
            
    def preprocess_image(self, images):
        return self.processor(images = images, return_tensors = "pt")["pixel_values"][0]
    
    def tokenize(self, text):
        if self.is_inference:
            kwargs = {}
        else:
            kwargs = {"padding" : "max_length", "max_length" : self.max_length, "truncation" : "True"}
        return self.processor.tokenizer(text = text, return_tensors = "pt", **kwargs)
    
    def __len__(self) -> int:
        return len(self.loaded_dataset)
    
    def _get_item_train(self, index):
        row = self.loaded_dataset[index]
        
        image_path = os.path.join(
            self.dataset_root, "MSCOCO/train/" + row["image"]
        )
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        images = [image]
        
        prompt = ""
        language = "value"
        for c in row["conversations"]:
            agent = c["from"]
            message = c[language]
            prompt += f"##{agent}: {message}\n:"
            
        tokenized = self.tokenize(prompt)
        tokenized_prompt = tokenized["input_ids"][0]
        labels = torch.full_like(tokenized_prompt, IGNORE_INDEX)
        prompt_attn_mask = tokenized["attention_mask"][0]
        
        index_ignore_loss = prompt_attn_mask.sum().item() + 1
        labels[:index_ignore_loss] = tokenized_prompt[:index_ignore_loss]
        
        return_dict = {
            "input_ids" : tokenized_prompt,
            "labels" : labels,
            "attention_mask" : prompt_attn_mask,
            "pixel_values" : self.preprocess_image(images),
        }
        return return_dict
    
    def _get_item_inference(self, index):
        row = self.loaded_dataset[index]
        
        image_path = os.path.join(
            self.dataset_root, "MSCOCO/train/" + row["image"]
        )
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        images = [image]
        
        language = "value"
        prompt = f"##human: {row['conversations'][language]}\n##gpt: "
        
        tokenized = self.tokenize(prompt)
        tokenized_prompt = tokenized["input_ids"][0]
        prompt_attn_mask = tokenized["attention_mask"][0]
        
        return_dict = {
            "input_ids" : tokenized_prompt,
            "labels" : tokenized_prompt,
            "attention_mask" : prompt_attn_mask,
            "pixel_values" : self.preprocess_image(images),
            "image" : images[0],
            "conversations" : row["conversations"],
            "prompt" : prompt,
        }
        
        return return_dict