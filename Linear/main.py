# 1. 선형회귀 모형 만들기
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import argparse

from layer import LinearLayer
from model import LinearRegression
from preprocess import *
from eval import evaluate
from train import train

def my_argparse():
    parser = argparse.ArgumentParser(description="Linear Model")

    parser.add_argument("-s",type=int, default = [11], nargs = 1, help="seed")
    parser.add_argument("-e", type=int, default = [100],nargs = 1, help="epoch")
    parser.add_argument("-b", type=int, default = [32],nargs = 1, help="batch_size")
    parser.add_argument("-lr", type=float,default = [1e-4], nargs = 1, help="learning_rate")
    
    args = parser.parse_args()
    return args.s[0], args.e[0], args.b[0], args.lr[0]


def main():

    seed, epoch, batch_size, learning_rate = my_argparse()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    x,y = data_generation(1000,4,seed)
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_split(x,y,seed)
    
    tr_dataset = CustomDataset(x_train, y_train)
    va_dataset = CustomDataset(x_valid, y_valid)
    te_dataset = CustomDataset(x_test, y_test)

    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
    va_dataloader = DataLoader(va_dataset, batch_size=batch_size, shuffle=True)
    te_dataloader = DataLoader(te_dataset, batch_size=batch_size, shuffle=True)

    model = LinearRegression()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for e in range(epoch+1):
        
        train_loss = train(model, tr_dataloader, criterion, optimizer, device)
        valid_loss = evaluate(model, va_dataloader, criterion, device)
        
        if e % 10 == 0:
            print(f'epoch = {e}\n\
                    train loss = {train_loss.item()}\n\
                    valid loss = {valid_loss.item()}')


if __name__ == "__main__":
    main()
