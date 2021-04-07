import os
import random
import glob
import numpy as np
import pandas as pd
import argparse
import torchvision.transforms as transforms

import torch
import torch.nn as nn
from importlib import import_module
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim import Adam
from MaskModules.dataset import MaskDataset

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_img_paths(img_paths):
    res = []
    for img_path in img_paths:
        res.extend(glob.glob(os.path.join(img_path, "*.*g")))
    return res

def train(args):
    model_dir = args.model_dir
    data_dir = args.data_dir
    # Check if Model Dir exists
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # Transformation: Resize & Normalize
    transform = transforms.Compose([
        transforms.Resize((224, 224), Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    ])

    # Create Train & Validation Dataset (Train : Validation = 9 : 1)
    img_paths = glob.glob(os.path.join(data_dir, '*'))
    print(len(img_paths))
    train_paths = img_paths[:-len(img_paths)//10]
    valid_paths = img_paths[-len(img_paths)//10:]

    train_dataset = MaskDataset(get_img_paths(train_paths), transform=transform)
    valid_dataset = MaskDataset(get_img_paths(valid_paths), transform=transform)

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    valid_loader=  DataLoader(
        valid_dataset,
        batch_size=args.val_batch_size,
        shuffle=True,
        num_workers=4
    )

    # Create Model
    model_module = getattr(import_module('MaskModules.model'), args.model)
    model = model_module(num_classes=18)
    
    # Set Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set Learning Rate
    lr = 1e-3

    # Set Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=lr)
    
    max_valid_acc = 0
    min_valid_loss = 100

    # Start Training
    print("Start Training...")
    for i in range(args.epochs):
        print("Epoch {} start...".format(i))
        model.train()
        loss_value = 0
        matches = 0
        for batch_idx, sample in enumerate(train_loader):
            inputs, targets = sample["image"].to(device), sample["label"].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            preds = outputs.argmax(axis=1)
            
            loss = criterion(outputs, targets.argmax(axis=1))
            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()

            if (batch_idx + 1) % args.batch_size:
                print(
                    f"Epoch[{epoch}/{args.epochs}]({batch_idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%}"
                )
        
        # Test
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = sample["image"].to(device), sample["label"].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets.argmax(axis=1))
                
                test_loss += loss.item()
                predicted = outputs.argmax(axis=1)
                total += targets.size(0)
                correct += predicted.eq(targets.argmax(axis=1)).sum().item()
            #TODO: save max valid accuracy model
        print(f'Accuracy of the network on the {total} test images: %d %%' % (100 * correct / total))
    # Create Save Directory
    save_dir_name = "{}_epoch_{}".format(args.model, args.epochs)
    if not os.path.exists(os.path.join(model_dir, save_dir_name)):
        os.mkdir(os.path.join(model_dir, save_dir_name))
    torch.save(os.path.join(model_dir, save_dir_name, 'best.pth'))


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--model_dir', type=str, default='./model', help='directory to save model')
    parser.add_argument('--data_dir', type=str, default='/opt/ml/input/data/train/images/', help='directory to save model')
    parser.add_argument('--model', type=str, default='ResNet_Model', help='train model (default: ResNet_Model)')
    parser.add_argument('--epochs', type=int, default=20, help='epochs (default: 20)')
    parser.add_argument('--augmentation', type=str, default=None) #TODO
    parser.add_argument('--batch_size', type=int, default=32, help='number of batch (default: 32)')
    parser.add_argument('--val_batch_size', type=int, default=32, help='number of validation batch (default: 32)')

    args = parser.parse_args()
    print(args)
    seed_everything(args.seed)
    train(args)