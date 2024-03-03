import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

def load_CIFAR10():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = CIFAR10(
        root = '../data/cifar10', 
        train = True, 
        download = True, 
        transform = transform
        )

    validset = CIFAR10(
        root = '../data/cifar10_test', 
        train = False, 
        download = True, 
        transform = transform
        )

    return trainset, validset
