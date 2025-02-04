import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import warnings
warnings.filterwarnings('ignore')

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')


class SegmentationDataset(Dataset):
    def __init__(self, transform, base_dir, images, masks):
        self.transform = transform
        self.images_path = os.path.join(base_dir, images)
        self.masks_path = os.path.join(base_dir, masks)
        self.len = len(os.listdir(self.images_path))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_path, f"img_{idx}.png")
        mask_path = os.path.join(self.masks_path, f"img_{idx}.png")
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        mask = None
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('RGB')
            if self.transform:
                mask = self.transform(mask)

        if mask is None:
            return image, f"img_{idx}.png"
        return image, mask