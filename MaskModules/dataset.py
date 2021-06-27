import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MaskDataset:
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform
        self.labels = []
        
        for img_path in img_paths:
            _, gender_str, _, age_str = img_path.split("/")[-2].split("_")
        
            mask_str = img_path.split("/")[-1]
            mask, sex, age = (0, 0, 0)

            if mask_str.startswith("incorrect"):
                mask = 1
            elif mask_str.startswith("normal"):
                mask = 2

            if gender_str == "female":
                sex = 1

            if int(age_str) >= 59:
                age = 2
            elif int(age_str) >= 30:
                age = 1
            self.labels.append((mask, sex, age))
    
    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        mask, sex, age = self.labels[index]
        label = [0 for _ in range(18)]
        label[mask * 6 + sex * 3 + age] = 1
        if self.transform:
            img = self.transform(img)
        return {"image": img, "label": mask*6+sex*3+age}
        
    def __len__(self):
        return len(self.img_paths)

