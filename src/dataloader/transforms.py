import torch
from random import uniform 
from torchvision.transforms import Compose,Normalize,RandomCrop,RandomResizedCrop,Resize,RandomHorizontalFlip, ToTensor
from torchvision import transforms


def get_transforms():

    # Normalization
    normalize = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    transform = Compose([normalize])

    # # Color augmentation
    # # Randomly choosing the intensity for augmentation
    # b = uniform(0, 0.5)
    # s = uniform(0, 0.5)
    # c = uniform(0, 0.5)

    # jitter = transforms.ColorJitter(brightness=b, contrast= c, saturation= s)
    # # randSharp = transforms.RandomAdjustSharpness(0.8, p=0.5)
    # transform = Compose([normalize,
    #                      jitter
    #                      ])
    return transform

