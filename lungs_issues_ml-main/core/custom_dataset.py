import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import warnings
from torchvision.transforms.v2 import functional as f
warnings.filterwarnings('ignore')

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomDataset(Dataset):
    def __init__(self, img_dir, annotations_file=None, transform=None, resize=256, 
                 brightness=1, contrast=1, sharpness=1, equalize=False, invert=False,
                 segmentation=False):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = pd.read_csv(annotations_file) if annotations_file is not None else None
        self.resize = resize 
        self.brightness = brightness
        self.contrast = contrast
        self.sharpness = sharpness
        self.equalize = equalize
        self.invert = invert

    def __len__(self):
        return len(self.img_labels) if self.img_labels is not None else len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        if self.img_labels is None:
            img_path = os.path.join(self.img_dir, f"img_{idx}.png")
        else:
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx][0])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        image = self.augmentation(image)
        if self.img_labels is None:
            return image
        return image, self.img_labels.iloc[idx][1]
    
    def augmentation(self, x):
        x = f.resize(x, self.resize)
        if self.invert:
            x = f.invert(x)
        if self.equalize:
            x = f.equalize(x)
        x = f.adjust_brightness(x, self.brightness)
        x = f.adjust_contrast(x, self.contrast)
        x = f.adjust_sharpness(x, self.sharpness)
        return x
    