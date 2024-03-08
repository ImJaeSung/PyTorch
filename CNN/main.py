import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import os
import argparse

from model import CNN
from preprocess import load_data, loader
from train import train
from eval import eval
from valid import valid


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def argparse_custom():
    parser = argparse.ArgumentParser(description="CNN")

    parser.add_argument("-s", type=int, default=[42], nargs=1, help="seed")
    parser.add_argument("-e", type=int, default=[100], nargs=1, help="epoch")
    parser.add_argument("-b", type=int, default=[64], nargs=1, help="batch_size")
    parser.add_argument("-lr", type=float, default=[1e-2], nargs=1, help="learning_rate")
    parser.add_argument("-es", type=int, default=[5], nargs=1, help="early_stopping")
    parser.add_argument("-esc", type=int, default=[0], nargs=1, help="early_stopping_count")
    parser.add_argument("-bvl", type=float, default=[float('inf')], nargs=1, help="best_valid_loss")
    
    args = parser.parse_args()

    args.s = args.s[0]
    args.e = args.e[0]
    args.b = args.b[0]
    args.lr = args.lr[0]
    args.es = args.es[0]
    args.esc = args.esc[0]
    args.bvl = args.bvl[0]

    return args


def main():

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(device)

    args = argparse_custom()
    seed = args.s
    epochs = args.e
    batch_size = args.b
    learning_rate = args.lr
    early_stopping = args.es
    early_stopping_count = args.esc
    best_valid_loss = args.bvl

    seed_everything(seed)

    train_set, valid_set, test_set = load_data(seed)
    train_loader, valid_loader, test_loader = loader(train_set, valid_set, test_set, batch_size)

    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)

    with tqdm(range(1, epochs+1)) as tr:
        for epoch in tr:

            train_loss = train(model, train_loader, optimizer, criterion, device)
            valid_loss = valid(model, valid_loader, criterion, device)

            if epoch % 5 == 0:
                print(f'epoch:{epoch}, train_loss:{train_loss.item():5f}')
                print(f'epoch:{epoch}, valid_loss:{valid_loss.item():5f}')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'best_model.pth')
                early_stopping_count = 0
            else:
                early_stopping_count += 1
            
            # 조기 종료 체크
            if early_stopping_count >= early_stopping:
                print('Stop early.')
                print(f'epoch:{epoch}, train_loss:{train_loss.item():5f}')
                print(f'epoch:{epoch}, valid_loss:{valid_loss.item():5f}')
                break

    model = CNN().to(device)
    model.load_state_dict(torch.load('best_model.pth'))

    accuracy = eval(model, test_loader, device)

    print(f'Accuracy on test data: {100 * accuracy:.2f}%')

if __name__ == "__main__":
     main()
