import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import random

class GaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
        
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class MultiplyNoise(object):
    def __init__(self, factor_range=(0.8, 1.2)):
        self.factor_range = factor_range
        
    def __call__(self, tensor):
        factor = random.uniform(self.factor_range[0], self.factor_range[1])
        return tensor * factor
        
    def __repr__(self):
        return self.__class__.__name__ + '(factor_range={0})'.format(self.factor_range)


class ConcreteCrackDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, stage="raw", transform=None, is_val=False):

        self.data_dir = data_dir
        self.stage = stage
        self.transform = transform
        self.is_val = is_val
        
        if is_val:
            self.data_path = os.path.join(data_dir, "CLIP_label", "valid")
        elif stage == "raw":
            self.data_path = os.path.join(data_dir, "CLIP_label", "train")
        elif stage == "aug":
            self.data_path = os.path.join(data_dir, "CLIP_label", "aug_train")
        else:
            raise ValueError("stage必须为'raw'或'aug'")
            
        self.crack_path = os.path.join(self.data_path, "crack")
        self.no_crack_path = os.path.join(self.data_path, "no_crack")
        
        if not os.path.exists(self.crack_path) or not os.path.exists(self.no_crack_path):
            raise FileNotFoundError(f"数据目录不存在: {self.data_path}")
            
        self.crack_images = [os.path.join(self.crack_path, f) for f in os.listdir(self.crack_path) if f.endswith('.jpg')]
        self.no_crack_images = [os.path.join(self.no_crack_path, f) for f in os.listdir(self.no_crack_path) if f.endswith('.jpg')]
        
        self.images = self.crack_images + self.no_crack_images
        
        self.labels = [1] * len(self.crack_images) + [0] * len(self.no_crack_images)
        
        dataset_type = "验证集" if is_val else "训练集"
        print(f"{dataset_type}加载了{len(self.images)}张图片，其中有裂缝{len(self.crack_images)}张，无裂缝{len(self.no_crack_images)}张")
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=3):

    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="训练"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())
        
        print(f'训练 Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="验证"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)
        val_losses.append(epoch_loss)
        val_accs.append(epoch_acc.item())
        
        print(f'验证 Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        scheduler.step()
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), f'../resnet50_model/resnet50_{args.stage}_best.pth')
            
        snapshot_path = f'../resnet50_model/resnet50_{args.stage}_epoch{epoch + 1}.pth'
        torch.save(model.state_dict(), snapshot_path)
        print(f"→ 保存周期性模型至 {snapshot_path}")
            
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='train')
    plt.plot(val_accs, label='val')
    plt.title('Accuracy')
    plt.legend()
    
    plt.savefig(f'../resnet50_model/resnet50_{args.stage}_training_curve.png')
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=str, default='aug', choices=['raw', 'aug'], help='训练阶段：raw或aug')
    parser.add_argument('--data_dir', type=str, default='../data/ConcreteData', help='数据目录')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"使用设备: {args.device}")
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            GaussianNoise(0., 0.01),
            MultiplyNoise((0.9, 1.1))
        ]),
        'val': transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    try:
        train_dataset = ConcreteCrackDataset(args.data_dir, stage=args.stage, transform=data_transforms['train'])
        val_dataset = ConcreteCrackDataset(args.data_dir, stage="raw", transform=data_transforms['val'], is_val=True)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    os.makedirs('../resnet50_model', exist_ok=True)
    
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=args.num_epochs)
    
    torch.save(model.state_dict(), f'../resnet50_model/resnet50_{args.stage}_final.pth')
    print("训练完成，模型已保存!")