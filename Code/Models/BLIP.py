# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 21:05:49 2024

@author: Nova18
"""

import cv2
import numpy as np
from datasets import load_dataset
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering, BlipImageProcessor, AutoProcessor
from transformers import BlipConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import display

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


# Configuration Base
config = BlipConfig.from_pretrained("Salesforce/blip-vqa-base")
class CFG:
    image_path = data_folder + 'MSCOCO/train2014'


# 2. Build Dataloader
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, data, text_processor, image_processor):
        self.data = data
        self.image_filenames = data['image']
        self.questions = data['question']
        self.answers = data['answer']
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.max_length = 32
        self.image_height = 224
        self.image_width = 224

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # get image + text
        answers = self.answers[idx]
        questions = self.questions[idx]
        text = self.questions[idx]
        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_encoding = self.image_processor(image,
                                              do_resize = True,
                                              size = (self.image_height, self.image_width),
                                              return_tensors = 'pt')
        
        text_encoding = self.text_processor(
                                            None,
                                            text,
                                            padding = 'max_length',
                                            truncation = True,
                                            max_length = self.max_length,
                                            return_tensors = 'pt'
                                            )
        
        # # remove batch dimension
        for k,v in text_encoding.items():
            text_encoding[k] = v.squeeze()
        text_encoding['pixel_values'] = image_encoding['pixel_values'][0]
        # # add labels
        labels = self.text_processor.tokenizer.encode(
            answers,
            max_length = self.max_length,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt"
        )[0]
        text_encoding['labels'] = labels
        
        return text_encoding
    
# Text & Image Process
text_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
image_processor = BlipImageProcessor.from_pretrained("Salesforce/blip-vqa-base")

train_vqa_dataset = VQADataset(data = train_df,
                               text_processor = text_processor, 
                               image_processor = image_processor)

val_vqa_dataset = VQADataset(data = test_df,
                               text_processor = text_processor, 
                               image_processor = image_processor)

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
                               batch_size = 64,
                               shuffle = False)

val_dataloader = DataLoader(val_vqa_dataset,
                               collate_fn = collate_fn,
                               batch_size = 64,
                               shuffle = False)

batch = next(iter(train_dataloader))
for k, v in batch.items():
    print(k, v.shape)
                              
# 4. Define the model
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base" )
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5)
image_mean = image_processor.image_mean
image_std = image_processor.image_std

# Test a sample
batch_idx = 1

unnormalized_image = (batch["pixel_values"][batch_idx].cpu().numpy() * np.array(image_std)[:, None, None]) + np.array(image_mean)[:, None, None]
unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
unnormalized_image =(unnormalized_image * 255).astype(np.uint8)
print("Question: ", text_processor.decode(batch["input_ids"][batch_idx]))
print("Answer: ", text_processor.decode(batch["labels"][batch_idx]))
plt.imshow(Image.fromarray(unnormalized_image))

# 5. Model Training
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

# 6. Inference
# add batch dimension + move to GPU
for x in range(10):
    sample = val_vqa_dataset[x]
    print("Question: ", text_processor.decode(sample['input_ids'], skip_special_tokens = True))
    sample = {k: v.unsqueeze(0).to(device) for k,v in sample.items()}
    
    # Forward pass
    outputs = model.generate(pixel_values = sample['pixel_values'],
                             input_ids = sample['input_ids'])
    print("Predicted Answer: ", text_processor.decode(outputs[0], skip_special_tokens = True))
    print("Actual Answer: ", text_processor.decode(sample['labels'][0], skip_special_tokens = True))
    #########################################################################
    unnormalized_image = (sample["pixel_values"][0].cpu().numpy() * np.array(image_std)[:, None, None]) + np.array(image_mean)[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    unnormalized_image = np.moveaxis(unnormalized_image, 0 , -1)
    display(Image.fromarray(unnormalized_image))
    #########################################################################
    print("#########################################################################")
