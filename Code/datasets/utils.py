#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 00:56:04 2024

@author: lurker18
"""

from typing import Dict, Tuple

import datasets
import pandas as pd
import yaml
from torch.utils.data import ConcatDataset, Dataset

from Code.Models.prepare_preprocessors import get_processor
from Code.datasets.llava_datasets import LlavaDataset
from Code.datasets.llava_instruct_datasets import LlavaInstructDataset
from Code.datasets.m3it_datasets import M3ITDataset
from Code.datasets.m3it_instruct_datasets import M3ITInstructDataset

dataset_classes = {
    "llava": LlavaDataset,
    "llava_instruct": LlavaInstructDataset,
    "m3it": M3ITDataset,
    "m3it_instruct": M3ITInstructDataset,
}


def get_each_dataset(dataset_config: Dict, processor, max_length: int) -> Tuple[Dataset, Dataset]:
    dataset_type = dataset_config["dataset_type"]
    if dataset_type not in dataset_classes:
        raise ValueError(f"dataset_type: {dataset_type} is not supported.")

    DatasetClass = dataset_classes[dataset_type]
    train_dataset = DatasetClass.create(dataset_config, processor, max_length, "train")
    val_dataset = DatasetClass.create(dataset_config, processor, max_length, "validation")
    return train_dataset, val_dataset


def get_dataset(config: Dict) -> Tuple[Dataset, Dataset]:
    processor = get_processor(config["model_config"])
    train_dataset_list = []
    val_dataset_list = []
    max_length = config["model_config"]["max_length"]

    for dataset_config_path in config["dataset_config_path"]:
        with open(dataset_config_path, "r") as f:
            dataset_config = yaml.safe_load(f)
        train_dataset, val_dataset = get_each_dataset(dataset_config, processor, max_length)
        train_dataset_list.append(train_dataset)
        val_dataset_list.append(val_dataset)

    train_dataset = ConcatDataset(train_dataset_list)
    val_dataset = ConcatDataset(val_dataset_list)

    return train_dataset, val_dataset