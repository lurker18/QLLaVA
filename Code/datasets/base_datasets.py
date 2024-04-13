#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 01:08:47 2024

@author: lurker18
"""

import abc
import traceback

from torch.utils.data import Dataset

IGNORE_INDEX = -100


class BaseDataset(Dataset):
    def __init__(self, is_inference: bool = False):
        super(BaseDataset, self).__init__()
        self.is_inference = is_inference
        
        
    @classmethod
    @abc.abstractmethod
    def create(cls, *args, **kwargs):
        raise NotImplementedError
        
    def __getitem__(self, index):
        if self.is_inference:
            return self._get_item_inference(index)
        else:
            return self._get_item_train(index)
        
    
    @abc.abstractmethod
    def _get_item_train(self, index):
        raise NotImplementedError
        
        
    @abc.abstractmethod
    def _get_item_inference(self, index):
        raise NotImplementedError
        
        
    
class ResilientDataset(BaseDataset):
    def __init__(self, is_inference: bool = False, max_trials: int = 5):
        super().__init__(is_inference)
        self.max_trials = max_trials
        
    def __getitem__(self, index: int):
        if self.is_inference:
            return self._get_item_inference(index)
        else:
            for _ in range(self.max_trials):
                try:
                    return self._get_item_train(index)
                except Exception as e:
                    print("Exception in ResilientDataset", e)
                    traceback.print_exc()
                    index += 1