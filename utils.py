import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Albumentations transform wrapper
class AlbumentationsTransform:
    def __init__(self, aug: A.Compose):
        self.aug = aug

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image_np = np.array(image)
        augmented = self.aug(image=image_np)

        return augmented["image"]

# Augmentations
def get_transforms():
    train_transform = A.Compose([
                
        # Transformations
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
        A.GaussNoise(),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.8),
        
        # Normalization + convert to PyTorch tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform
    }

# Dataset Class
class PokemonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label    

# Model
class PokemonResNet(nn.Module):
        def __init__(self, num_classes):
            super(PokemonResNet, self).__init__()
            self.base_model = models.resnet18(pretrained=True)
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(in_features, num_classes)

        def forward(self, x):
            x = self.base_model(x)
            return x

albumentations_transforms = get_transforms()

# Apply augmentation
train_augs = AlbumentationsTransform(albumentations_transforms["train"])
val_augs   = AlbumentationsTransform(albumentations_transforms["val"])
test_augs  = AlbumentationsTransform(albumentations_transforms["test"])

# Datasets
train_dataset = PokemonDataset("data/pokemon-dataset-1000/train", transform=train_augs)
val_dataset   = PokemonDataset("data/pokemon-dataset-1000/val",   transform=val_augs)
test_dataset  = PokemonDataset("data/pokemon-dataset-1000/test",  transform=test_augs)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=2)