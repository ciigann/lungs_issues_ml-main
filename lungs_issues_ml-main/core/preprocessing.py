from torchvision.transforms import transforms
import torch
from torchvision.transforms.v2 import functional as f

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_train_transofrms(grayscale=False, flip_prob=0, rot_angle=0):
    list_transforms = [
        transforms.RandomHorizontalFlip(flip_prob),
        transforms.RandomRotation(rot_angle),
        transforms.ToTensor()
    ]
    if grayscale:
        list_transforms.append(transforms.Grayscale())
    return transforms.Compose(list_transforms)


def get_test_transforms(grayscale=False):
    list_transforms = [
        transforms.ToTensor()
    ]
    if grayscale:
        list_transforms.append(transforms.Grayscale())
    return transforms.Compose(list_transforms)
