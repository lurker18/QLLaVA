# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 01:20:44 2024

@author: Nova18
"""

import cv2
import numpy as np
from datasets import load_dataset
from PIL import Image
import torch
from transformers import  GitForCausalLM, GitProcessor
from transformers import GitConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model
from IPython.display import display
from huggingface_hub import hf_hub_download

from utils import data_preprocessing

data_folder = 'C:/Users/Nova18/Desktop/MLLM/data/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Dataset
llava_150k = load_dataset('json', data_files = data_folder + 'llava_instruct_150k.json')
#llava_conv = load_dataset('json', data_files = data_folder + 'conversation_58k.json')
#llava_details = load_dataset('json', data_files = data_folder + 'detail_23k.json')
#llava_reasoning = load_dataset('json', data_files = data_folder + 'complex_reasoning_77k.json')

### Path-VQA ###
#pathvqa = load_dataset("flaviagiammarino/path-vqa")

df = data_preprocessing(llava_150k)
df.info()

# Train / Test split
train, test = train_test_split(df, test_size = 0.2)
train_df = train.reset_index()
test_df = test.reset_index()
del train_df['index']
del test_df['index']
########################################################################################################
# Configuration Base
config = GitConfig.from_pretrained("microsoft/git-base")
class CFG:
    image_path = data_folder + 'MSCOCO/train2014'


# 2. Build Dataloader
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.image_filenames = data['image']
        self.questions = data['question']
        self.answers = data['answer']
        self.processor = processor
        self.max_length = 32

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # get image + text
        answers = self.answers[idx]
        questions = self.questions[idx]
        image = Image.open(f"{CFG.image_path}/{self.image_filenames[idx]}").convert("RGB")
        #image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Image for pixel_values
        encoding = self.processor(images = image, 
                                  text = questions,  
                                  max_length = self.max_length,
                                  padding = "max_length", 
                                  truncation = True,
                                  return_tensors = "pt")
        

        # # remove batch dimension
        for k,v in encoding.items():
            encoding[k] = v.squeeze()
        # # add labels
        labels = self.processor.tokenizer.encode(
            answers,
            max_length = self.max_length,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt"
        )[0]
        encoding['labels'] = labels
        
        return encoding
    
# Text & Image Process
processor = GitProcessor.from_pretrained("microsoft/git-base-textvqa")


train_vqa_dataset = VQADataset(data = train_df,
                               processor = processor)

val_vqa_dataset = VQADataset(data = test_df,
                               processor = processor)

# Test some sample    
train_vqa_dataset[0]
val_vqa_dataset[0]

# 3. Join Image and Texts together
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    # create new batch
    batch = {}
    batch['input_ids'] = torch.stack(input_ids)
    batch['attention_mask'] = torch.stack(attention_mask)
    batch['pixel_values'] = torch.stack(pixel_values)
    batch['labels'] = torch.stack(labels)
    
    return batch

train_dataloader = DataLoader(train_vqa_dataset,
                              collate_fn = collate_fn,
                              batch_size = 8,
                              shuffle = False)

val_dataloader = DataLoader(val_vqa_dataset,
                            collate_fn = collate_fn,
                            batch_size = 8,
                            shuffle = False)

batch = next(iter(train_dataloader))
for k, v in batch.items():
    print(k, v.shape)
                              
# 4. Define the model
model = GitForCausalLM.from_pretrained("microsoft/git-base-textvqa", load_in_8bit = True)

# 5. Set LoRA configuration
config = LoraConfig(
    r = 16,
    lora_alpha = 32,
    lora_dropout = 0.05,
    bias = "none",
    target_modules = ['q_proj', 'k_proj']
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5)
image_mean = [0.48145466, 0.4578275, 0.40821073]
image_std = [0.26862954, 0.26130258, 0.27577711]

# Test a sample
batch_idx = 1

unnormalized_image = (batch["pixel_values"][batch_idx].cpu().numpy() * np.array(image_std)[:, None, None]) + np.array(image_mean)[:, None, None]
unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
unnormalized_image =(unnormalized_image * 255).astype(np.uint8)
print("Question: ", processor.decode(batch["input_ids"][batch_idx]))
print("Answer: ", processor.decode(batch["labels"][batch_idx]))
plt.imshow(Image.fromarray(unnormalized_image))

# 6. Model Training
model.train()
for epoch in range(5):
    print(f"Epoch: {epoch}")
    total_loss = []
    for batch in tqdm(train_dataloader):
        # get the inputs;
        batch = {k:v.to(device) for k,v in batch.items()}
        
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(**batch)
        loss = outputs.loss
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    print("Loss:", sum(total_loss))


# 7. Inference
# add batch dimension + move to GPU
for x in range(10):
    sample = val_vqa_dataset[x]
    print("Question: ", processor.decode(sample['input_ids'], skip_special_tokens = True))
    sample = {k: v.unsqueeze(0).to(device) for k,v in sample.items()}
    
    # Forward pass
    outputs = model.generate(pixel_values = sample['pixel_values'],
                             input_ids = sample['input_ids'])
    print("Predicted Answer: ", processor.decode(outputs[0], skip_special_tokens = True))
    print("Actual Answer: ", processor.decode(sample['labels'][0], skip_special_tokens = True))
    #########################################################################
    unnormalized_image = (sample["pixel_values"][0].cpu().numpy() * np.array(image_std)[:, None, None]) + np.array(image_mean)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0 , -1)
    display(Image.fromarray(unnormalized_image))
    #########################################################################
    print("#########################################################################")

