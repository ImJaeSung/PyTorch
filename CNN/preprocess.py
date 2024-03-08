import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def load_data(seed):

    train_set = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                            train=True, # True를 지정하면 훈련 데이터로 다운로드
                            transform=transforms.ToTensor(), # 텐서로 변환
                            download=True)

    test_set = dsets.MNIST(root='MNIST_data/', # 다운로드 경로 지정
                            train=False, # False를 지정하면 테스트 데이터로 다운로드
                            transform=transforms.ToTensor(), # 텐서로 변환
                            download=True)

    train_set, valid_set = random_split(train_set, [len(train_set)-len(test_set), len(test_set)], torch.Generator().manual_seed(seed))

    return train_set, valid_set, test_set

def loader(train_set, valid_set, test_set, batch_size=64):

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, valid_loader, test_loader