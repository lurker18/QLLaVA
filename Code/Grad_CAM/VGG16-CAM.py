# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 00:02:22 2024

@author: Nova18
"""

import os
import cv2
import time
import random
from random import randint
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

random_seed = 42

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

np.random.seed(random_seed)
random.seed(random_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# SETUP Folders
data_folder = 'C:/Users/Nova18/Desktop/MLLM/data'
results_folder = 'C:/Users/Nova18/Desktop/MLLM/results'
# Enter the keyword of the disease:
types_disease = 'pneumonia'

# 1. Preprocessing of image type data # 
class CTScanDataset(Dataset):
    def __init__(self, directory, types_disease, transform = None, train = True, test = False):
        self.images, self.labels = self.get_images(types_disease, directory)
        self.types_disease = types_disease
        self.directory = directory
        self.train = train
        self.test = test
        self.transform = transform
        
        if self.transform is None:
            if types_disease == 'covid19' or types_disease == 'COVID19': # for COVID19 disease
                norm_mean = [0, 0, 0]
                norm_std = [1, 1, 1]

            else: # for pneumonia disease
                norm_mean = [0.485, 0.456, 0.406]
                norm_std = [0.229, 0.224, 0.225]
            
            normalize = T.Normalize(norm_mean, norm_std)
            
            if test or not train: # test set
                if types_disease == 'covid19' or types_disease == 'COVID19':
                    self.transform = T.Compose([
                        T.Resize(225),
                        T.CenterCrop(225),
                        T.ToTensor(),
                        normalize 
                        #normalize
                    ])
                else:
                    self.transform = T.Compose([
                        T.Resize(225),
                        T.CenterCrop(225),
                        T.ToTensor(),
                        #normalize
                    ])
            else: # train set
                if types_disease == 'covid19' or types_disease == 'COVID19':
                    self.transform = T.Compose([
                        T.Resize(256),
                        T.CenterCrop(256),
                        T.RandomCrop(225),
                        T.RandomHorizontalFlip(p = 0.5),
                        T.ToTensor(),
                        normalize 
                        #normalize
                    ])
                else:
                    self.transform = T.Compose([
                        T.Resize(256),
                        T.CenterCrop(256),
                        T.RandomCrop(225),
                        T.RandomHorizontalFlip(p = 0.5),
                        T.ToTensor(),
                        #normalize
                    ])
                
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        # Convert to PIL Image
        image = Image.fromarray(image)
        
        # Apply transformations
        image = self.transform(image)
        
        # Convert to PyTorch tensor
        image = torch.tensor(image, dtype = torch.float32)
        label = torch.tensor(label, dtype = torch.long)
        return image, label
    
    
    def get_images(self, types_disease, directory):
        images = []
        labels = []
        if types_disease == 'covid19' or types_disease == 'COVID19':
            labels_mapping = ['COVID', 'NonCOVID']
        else:
            labels_mapping = ["NORMAL", "PNEUMONIA"]

        for label in labels_mapping:
            path = os.path.join(directory, label)
            class_num = labels_mapping.index(label)
            for img in os.listdir(path):
                image = cv2.imread(os.path.join(path, img))
                images.append(image)
                labels.append(class_num)
                
        return images, labels
    
def get_classlabels(types_disease, class_code):
    if types_disease == 'covid19' or types_disease == 'COVID19':
        labels = {0 : "NonCOVID",
                  1 : 'COVID'}
    else:
        labels = {0 : "Normal", 
                  1 : "Pneumonia"}
    return labels[class_code]

# 2. The distribution of data class #
def draw_hist(data_set, types_disease, title):
    labels = []
    
    if types_disease == 'covid19' or types_disease == 'COVID19':
        for _, label in data_set:
            for i in label:
                if i == 0:
                    labels.append("NonCOVID")
                else:
                    labels.append("COVID")
    else:
        for _, label in data_set:
            for i in label:
                if i == 0:
                    labels.append("Normal")
                else:
                    labels.append("Pneumonia")
                
    sns.set(style = "whitegrid", color_codes = True)
    if  types_disease == 'covid19' or types_disease == 'COVID19':
        ax = sns.countplot(x = labels, order = ['NonCOVID', "COVID"], palette = ['#432371',"#FAAE7B"])
        
    else:
        ax = sns.countplot(x = labels, order = ['Normal', "Pneumonia"], palette = ['#432371',"#FAAE7B"])
    
    # Add count annotations on top of each bar
    for p in ax.patches:
        ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'center', xytext = (0, 5), textcoords = 'offset points')
        
    plt.title(f"Class Distribution Histogram of {title}")
    plt.show()
    
# PATH for Train / Validation / Test sets
def choose_dataset(types_disease):
    if  types_disease == 'covid19' or types_disease == 'COVID19':
        train_path = data_folder + '/Covid_Dataset/train/'
        val_path = data_folder + '/Covid_Dataset/val/'
        test_path = data_folder + '/Covid_Dataset/test/'
    else:
        train_path = data_folder + '/chest_xray/train/'
        val_path = data_folder + '/chest_xray/val/'
        test_path = data_folder + '/chest_xray/test/'
    
    return train_path, val_path, test_path

train_path, val_path, test_path = choose_dataset(types_disease)

# Create instances of CustDAtaset for training / validation / testing
train_dataset = CTScanDataset(directory = train_path, types_disease = types_disease, train = True, test = False)
val_dataset = CTScanDataset(directory = val_path, types_disease = types_disease, train = False, test = True)
test_dataset = CTScanDataset(directory = test_path, types_disease = types_disease, train = False, test = True)

# Use DataLoader to create iterators for training / validation / testing
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

# Draw the histogram
#draw_hist(train_loader, types_disease, title = "Train Set")
#draw_hist(val_loader, types_disease, title = "Validation Set")
#draw_hist(test_loader, types_disease, title = "Test Set")


f, ax = plt.subplots(4, 4, figsize = (10, 10))
f.subplots_adjust(0, 0, 1, 1)
for i in range(4):
    for j in range(4):
        if i < 2:
            la = 0
        
        else:
            la = 1
            
        # randomly select image based on the la
        while True:
            rnd_number = randint(0, len(test_dataset) - 1)
            image, label = test_dataset[rnd_number]
            if label == la:
                break
            
        image = image.permute(1,2,0)
        ax[i,j].imshow(image)
        ax[i,j].set_title(get_classlabels(types_disease, label.item()))
        ax[i,j].axis('off')

plt.tight_layout()
plt.show()

# 3. Training of CNN #
def get_class_labels(types_disease):
    if  types_disease == 'covid19' or types_disease == 'COVID19':
        return ['COVID', 'NonCOVID']
    else:
        return ["Normal", "Pneumonia"]
    
#### Train
def train_and_evaluate(model, train_loader, test_loader, optimizer, model_name, num_epochs = 10, p = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)
    
    criterion = nn.CrossEntropyLoss(reduction = 'mean')
    train_loss_set, test_loss_set = [], []
    acc_train_set, acc_test_set = [], []
    best_test_loss = float('inf')
    
    train_start = time.time()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        train_acc_sum = 0.0
        all_train_labels, all_train_outputs = [], []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            all_train_labels.extend(labels.cpu().detach().numpy())
            outputs = outputs.argmax(dim = 1)
            all_train_outputs.extend(outputs.cpu().detach().numpy())
            
        average_train_loss = total_loss / len(train_loader)
        train_accuracy = accuracy_score(all_train_labels, all_train_outputs)
        train_loss_set.append(average_train_loss)
        acc_train_set.append(train_accuracy)
        
        model.eval()
        all_test_labels, all_test_preds = [], []
        with torch.no_grad():
            total_test_loss = 0.0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                _, preds = torch.max(outputs, 1) # pick the highest
                all_test_labels.extend(labels.cpu().numpy())
                all_test_preds.extend(preds.cpu().numpy())
                total_test_loss += criterion(outputs, labels).item()
                
        average_test_loss = total_test_loss / len(test_loader)
        test_loss_set.append(average_test_loss)
        
        test_accuracy = accuracy_score(all_test_labels, all_test_preds)
        acc_test_set.append(test_accuracy)
        current_lr = optimizer.param_groups[0]['lr']
        
        if average_test_loss < best_test_loss:
            # Save the model weights when the tetst loss is the lowest
            best_test_loss = average_test_loss
            torch.save(model.state_dict(), f"{results_folder}/model_weights_{model_name}.pth")
            
        if epoch % p == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: \t Lr: {current_lr},\t train loss: {average_train_loss:.4f},\t train acc: {train_accuracy:.4f},\t test loss:{average_test_loss:.4f},\t test acc:{test_accuracy:.4f} ")
            print("-" * 80)
            
    train_end = time.time()
    time_use = train_end - train_start
    print(f"Time used for Training:{time_use} sec ")
    print("-" * 80)
    
    report_train = classification_report(all_train_labels, all_train_outputs, target_names = get_class_labels(types_disease))
    report_test = classification_report(all_test_labels, all_test_preds, target_names = get_class_labels(types_disease))
    print(f"Training Classification Report :\n {report_train}")
    print(f"Test Classification Report :\n {report_test}")
    
    return model, train_loss_set, test_loss_set, acc_train_set, acc_test_set, time_use

# 4. Visualize of training process
def plot_graph(train_loss_history, test_loss_history, acc_train_loss, acc_test_loss, model_name):
    epochs = range(1, len(train_loss_history) + 1)
    
    fig, (ax1, ax2) = plt.subplots(2,1,figsize = (10, 10))
    
    ax1.plot(epochs, train_loss_history, label = 'Train Loss')
    ax1.plot(epochs, test_loss_history, label = 'Test Loss')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Training and test Loss Over Epochs on {model_name}")
    ax1.legend()
    
    ax2.plot(epochs, acc_train_loss, label = 'Train Acc')
    ax2.plot(epochs, acc_test_loss, label = "Test Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(F"Training and test Accuracy Over Epochs on {model_name}")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{results_folder}/epochs on {model_name}.jpg")
    plt.show()
    
# 5. Model Test
def test_model(model, test_loader, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)
    
    model.eval
    all_test_labels, all_test_outputs = [], []
    time_start = time.time()
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        _, preds = torch.max(outputs, 1)
        all_test_labels.extend(labels.cpu().numpy())
        all_test_outputs.extend(preds.cpu().numpy())
        
    results_df = pd.DataFrame({"True Labels" : all_test_labels, "Predicted Labels" : all_test_outputs})
    #results_df.to_csv(f"{results_folder}/CSV/{model_name}.csv", index = False)
    
    time_end = time.time()
    time_use = time_end - time_start
    report_test = classification_report(all_test_labels, all_test_outputs, target_names = get_class_labels(types_disease))
    conf_matrix = confusion_matrix(all_test_labels, all_test_outputs)
    print(f"Time use : {time_use} sec")
    print(f"Test Classification Report :\n {report_test}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    # Plot confusion matrix
    plt.figure(figsize = (8,6))
    sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = [0,1], yticklabels = [0,1])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(f"{results_folder}/{model_name}.jpg")
    plt.show()

# 6. CNN: VGG16 NET Pretrained
class VGG16(nn.Module):
    def __init__(self, num_classes = 2):
        super(VGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained = True)
        num_features = self.vgg16.classifier[0].in_features
        
        # Replace the classifier layers
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)  # Modify this layer based on your task
        )

    def forward(self, x):
        return self.vgg16(x)
    
vgg16 = VGG16()
model_name = "vgg16pre"
number_epochs = 30
p = 1
optimizer = optim.Adam(vgg16.parameters(), 
                       lr = 0.0001,
                       weight_decay = 0.001)
model_VGG16, train_loss, test_loss, acc_train, acc_test, time_use = train_and_evaluate(vgg16, 
                                                                                       train_loader, 
                                                                                       test_loader, 
                                                                                       optimizer = optimizer,
                                                                                       model_name = model_name,
                                                                                       num_epochs = number_epochs, 
                                                                                       p = p)

plot_graph(train_loss, test_loss, acc_train, acc_test, "VGG16_pretrained")

# Laod the saved parameters
loaded_vgg16_model = VGG16()
loaded_vgg16_model.to('cuda')
loaded_vgg16_model.load_state_dict(torch.load(results_folder + '/model_weights_vgg16pre.pth'))

features = loaded_vgg16_model.vgg16.features(test_dataset[0][0].unsqueeze(0).to('cuda'))
features.flatten()
loaded_vgg16_model.vgg16.classifier(features.flatten())

test_model(loaded_vgg16_model, test_loader,'VGG16 TEST')
test_model(loaded_vgg16_model, val_loader,'VGG16 VAL')

# 7. CAM: Class Activate Map:
def CAM_ALG(model, img):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Change img from test_dataset to 4D
    img = img.unsqueeze(0) # torch.Size([1, 3, 225, 225])
    img = img.to(device)
    
    # output of features and predict label
    model.eval()
    features = model.vgg16.features(img) # torch.Size([1, 256, 6, 6])
    output = model.vgg16.classifier(features.flatten())
    print(output)
    
    def extract(g):
        global features_grad
        features_grad = g
        
    pred_label = torch.argmax(output).item()
    pred_class = output[pred_label]
    
    features.register_hook(extract)
    pred_class.backward()
    
    grads = features_grad
    
    GAP_features = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
    
    Weighted_sum = features * GAP_features
    Weighted_sum = Weighted_sum.squeeze().cpu().detach().numpy()
    cam = np.mean(Weighted_sum, axis = 0)
    
    cam = np.maximum(cam, 0)

    cam /= np.maximum(np.max(cam), 0.00000000001)
    
    # Resize the CAM to the original image size and Min-Max Normalization
    cam = cv2.resize(cam, (225, 225))
    
    # Apply heatmap to the original image
    heatmap = cv2.applyColorMap(np.uint8(255 * (1 - cam)), cv2.COLORMAP_JET)
    img = np.transpose(img.squeeze().cpu().numpy(), (1, 2, 0))
    heatmap = heatmap / 255 # to normalize
    superimposed_img = heatmap * 0.4 + img
    
    return superimposed_img, pred_label

def CAM_visual(model, test_dataset, model_name, list_img):
    f, ax = plt.subplots(2, 4, figsize = (13, 8))
    
    for i, img_id in enumerate(list_img):
        img, label = test_dataset[img_id]
        
        image = img.permute(1, 2, 0)
        ax[0, i].imshow(image)
        ax[0, i].set_title(f"Label: {get_classlabels(types_disease, label.item())}")
        
        # img with CAM
        superimposed_img, predict_label = CAM_ALG(model, img)
        cas = ax[1, i].imshow(superimposed_img) 
        
        ax[1, i].set_title(f"Predict: {get_classlabels(types_disease, predict_label)}")
        
    plt.tight_layout()
    plt.subplots_adjust(top = 0.93)
    plt.suptitle(f"CAM visual On {model_name} model", fontsize = 16)
    plt.savefig(f"{results_folder}/{model_name}compareC.jpg")
    plt.show()
    
# Load the saved parameters
loaded_vgg16_model = VGG16()
loaded_vgg16_model.load_state_dict(torch.load(f'{results_folder}/model_weights_vgg16pre.pth'))    

# VGG16 without Disease
moedl_name = 'VGG16_NonDisease'
list_img = [10, 22, 63, 94]
CAM_visual(loaded_vgg16_model, test_dataset, model_name, list_img)


# VGG16 with Disease
model_name = 'VGG16_Disease'
list_img = [100, 169, 163, 194]
CAM_visual(loaded_vgg16_model, test_dataset, model_name, list_img)

loaded_vgg16_model
