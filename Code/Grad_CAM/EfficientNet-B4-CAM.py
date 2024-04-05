# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 00:44:49 2024

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
        
    results_df = pd.DataFrame({"True Labels" : all_test_labels, 
                               "Predicted Labels" : all_test_outputs})
    #results_df.to_csv(f"{results_folder}/CSV/{model_name}.csv", index = False)
    
    time_end = time.time()
    time_use = time_end - time_start
    report_test = classification_report(all_test_labels, all_test_outputs, target_names = get_class_labels(types_disease))
    conf_matrix = confusion_matrix(all_test_labels, all_test_outputs)
    print(f"Time use : {time_use} sec")
    print(f"Test Classification Report :\n {report_test}")
    print(f"Confusion Matrix:\n {conf_matrix}")
    
    # Plot confusion matrix
    plt.figure(figsize = (8,6))
    sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = [0,1], yticklabels = [0,1])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(f"{results_folder}/{model_name}.jpg")
    plt.show()


# 모델 설정 값

config = {
    # Classfier 설정
    "cls_hidden_dims" : []
    }

# 6. CNN: EfficientNetB4 Pretrained
class EFFICIENTNETB4(nn.Module):
    """pretrain 된 ResNet을 이용해 CT image embedding
    """
    
    def __init__(self):
        """
		Args:
			base_model : efficientnet_b0 / efficientnet_b4
			config: 모델 설정 값
		"""
        super(EFFICIENTNETB4, self).__init__()

        model = models.efficientnet_b4(pretrained = True)
        num_ftrs = model.classifier[-1].in_features
        self.num_ftrs = num_ftrs

        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        b = x.size(0)
        x = x.view(b, -1)

        return x
    
class Classifier(nn.Sequential):
    """임베딩 된 feature를 이용해 classificaion
    """
    def __init__(self, model_image, **config):
        """
        Args:
            model_image : image emedding 모델
            config: 모델 설정 값
        """
        super(Classifier, self).__init__()

        self.model_image = model_image # image 임베딩 모델

        self.input_dim = model_image.num_ftrs # image feature 사이즈
        self.dropout = nn.Dropout(0.1) # dropout 적용

        self.hidden_dims = config['cls_hidden_dims'] # classifier hidden dimensions
        layer_size = len(self.hidden_dims) + 1 # hidden layer 개수
        dims = [self.input_dim] + self.hidden_dims + [2] 

        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)]) # classifer layers 

    def forward(self, v):
        # Drug/protein 임베딩
        v_i = self.model_image(v) # batch_size x hidden_dim 

        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor)-1):
                # If last layer,
                v_i = l(v_i)
            else:
                # If Not last layer, dropout과 ReLU 적용
                v_i = F.relu(self.dropout(l(v_i)))

        return v_i
    
model_image = EFFICIENTNETB4()
efficientnetb4 = Classifier(model_image, **config)
model_name = "efficientnetb4pre"
number_epochs = 30
p = 1
optimizer = optim.Adam(efficientnetb4.parameters(), 
                       lr = 0.0001,
                       weight_decay = 0.001)
model_EFFICIENTNETB4, train_loss, test_loss, acc_train, acc_test, time_use = train_and_evaluate(efficientnetb4, 
                                                                                                train_loader, 
                                                                                                test_loader, 
                                                                                                optimizer = optimizer,
                                                                                                model_name = model_name,
                                                                                                num_epochs = number_epochs, 
                                                                                                p = p)

plot_graph(train_loss, test_loss, acc_train, acc_test, "EFFICIENTNETB4_pretrained")

# Laod the saved parameters
loaded_efficientnetb4_model = efficientnetb4
loaded_efficientnetb4_model.load_state_dict(torch.load(results_folder + '/model_weights_efficientnetb4pre.pth'))

test_model(loaded_efficientnetb4_model, test_loader,'EFFICIENTNETB4 TEST')
test_model(loaded_efficientnetb4_model, val_loader,'EFFICIENTNETB4 VAL')


def show_gradCAM(model, img):
    """gradCAM을 이용하여 활성화맵(activation map)을 이미지 위에 시각화하기
    args:
    model (torch.nn.module): 학습된 모델 인스턴스
    class_ind (int): 클래스 index [0 - NonCOVID, 1 - COVID]
    img: 시각화 할 입력 이미지
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Change img from test_dataset to 4D
    img = img.unsqueeze(0) # torch.Size([1, 3, 225, 225])
    img = img.to(device)
    
    model.eval()
    features = model.model_image.features(img)
    output = model.predictor[0](features.flatten())
    
    pred_label = torch.argmax(output).item()
    
    # target_layers = [model.layer4[-1]] # 출력층 이전 마지막 레이어 가져오기
    target_layers = [model.model_image.features[-2][-1]]
    cam = GradCAM(model = model,
                  target_layers = target_layers) 


    targets = [ClassifierOutputTarget(1)] # 타겟 지정
    grayscale_cam = cam(input_tensor = img, targets = targets)
    grayscale_cam = grayscale_cam[0, :]

    # 활성화맵을 이미지 위에 표시
    visualization = show_cam_on_image(img.squeeze(0).permute(1, 2, 0).cpu().numpy(), 
                                      grayscale_cam, 
                                      use_rgb = True) 

    pil_image = Image.fromarray(visualization)
    return pil_image, pred_label


def CAM_visual(model, test_dataset, model_name, list_img):
    f, ax = plt.subplots(2, 4, figsize = (13, 8))
    
    for i, img_id in enumerate(list_img):
        img, label = test_dataset[img_id]
        
        image = img.permute(1, 2, 0)
        ax[0, i].imshow(image)
        ax[0, i].set_title(f"Label: {get_classlabels(types_disease, label.item())}")
        
        # img with CAM
        superimposed_img, predict_label = show_gradCAM(model, img)
        cas = ax[1, i].imshow(superimposed_img) 
        
        ax[1, i].set_title(f"Predict: {get_classlabels(types_disease, predict_label)}")
        
    plt.tight_layout()
    plt.subplots_adjust(top = 0.93)
    plt.suptitle(f"CAM visual On {model_name} model", fontsize = 16)
    plt.savefig(f"{results_folder}/{model_name}compareC.jpg")
    plt.show()

    
# Load the saved parameters
loaded_efficientnetb4_model = efficientnetb4
loaded_efficientnetb4_model.load_state_dict(torch.load(f'{results_folder}/model_weights_efficientnetb4pre.pth'))    

# RESNET50 with NonCovid
moedl_name = 'EFFICIENTNETB4_NonCovid'
list_img = [10, 22, 20, 94]
CAM_visual(loaded_efficientnetb4_model, test_dataset, model_name, list_img)


# RESNET50 with COVID
model_name = 'EFFICIENTNETB4_Covid'
list_img = [140, 160, 200, 201]
CAM_visual(loaded_efficientnetb4_model, test_dataset, model_name, list_img)

