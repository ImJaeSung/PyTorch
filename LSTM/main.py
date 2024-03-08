import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from model import LSTM
from preprocess import *
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
    parser.add_argument("-e", type=int, default=[200], nargs=1, help="epoch")
    parser.add_argument("-b", type=int, default=[64], nargs=1, help="batch_size")
    parser.add_argument("-lr", type=float, default=[1e-2], nargs=1, help="learning_rate")
    parser.add_argument("-es", type=int, default=[15], nargs=1, help="early_stopping")
    parser.add_argument("-esc", type=int, default=[0], nargs=1, help="early_stopping_count")
    parser.add_argument("-bvl", type=float, default=[float('inf')], nargs=1, help="best_valid_loss")
    parser.add_argument("-iw", type=int, default=[24], nargs=1, help="input_window")
    parser.add_argument("-ow", type=int, default=[1], nargs=1, help="output_window")
    parser.add_argument("-hn", type=float, default=[64], nargs=1, help="hidden_size")
    parser.add_argument("-nl", type=int, default=[1], nargs=1, help="num_layers")
    
    args = parser.parse_args()

    args.s = args.s[0]
    args.e = args.e[0]
    args.b = args.b[0]
    args.lr = args.lr[0]
    args.es = args.es[0]
    args.esc = args.esc[0]
    args.bvl = args.bvl[0]
    args.iw = args.iw[0]
    args.ow = args.ow[0]
    args.hn = args.hn[0]
    args.nl = args.nl[0]

    return args

def main():

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    args = argparse_custom()
    seed = args.s
    epochs = args.e
    batch_size = args.b
    learning_rate = args.lr
    early_stopping = args.es
    early_stopping_count = args.esc
    best_valid_loss = args.bvl
    input_window = args.iw
    output_window = args.ow
    hidden_size = args.hn
    num_layers = args.nl

    seed_everything(seed)

    train_set, valid_set, test_set, test_true = preprocess(load_data())

    scaler = MinMaxScaler()
    train_set = scaler.fit_transform(train_set)
    valid_set = scaler.transform(valid_set)
    test_set = scaler.transform(test_set)

    test_final = test_set.copy()

    input_size = train_set.shape[1]
    output_size = output_window

    train_set = WindowDataset(train_set, input_window, output_window, input_size)
    valid_set = WindowDataset(valid_set, input_window, output_window, input_size)
    test_set = WindowDataset(test_set, input_window, output_window, input_size)

    train_loader, valid_loader, test_loader = loader(train_set, valid_set, test_set, batch_size=batch_size)

    model = LSTM(input_size, hidden_size, output_size, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss().to(device)

    with tqdm(range(1, epochs+1)) as tr:
        for epoch in tr:

            train_loss = train(model, train_loader, optimizer, criterion, device)
            valid_loss = valid(model, valid_loader, criterion, device)

            if epoch % 10 == 0:
                print(f'epoch:{epoch}, train_loss:{train_loss.item():5f}')
                print(f'epoch:{epoch}, valid_loss:{valid_loss.item():5f}')

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'best_lstm.pth')
                early_stopping_count = 0
            else:
                early_stopping_count += 1
            
            # 조기 종료 체크
            if early_stopping_count >= early_stopping:
                print(f'epoch:{epoch}, train_loss:{train_loss.item():5f}')
                print(f'epoch:{epoch}, valid_loss:{valid_loss.item():5f}')
                print(f'best valid loss :{best_valid_loss}')
                break

    model = LSTM(input_size, hidden_size, output_size, num_layers).to(device)
    model.load_state_dict(torch.load('best_lstm.pth'))

    predictions = eval(model, test_loader, device)
    test_final[24:,-1] = torch.tensor(predictions).cpu().numpy()
    final_pred = scaler.inverse_transform(test_final)[24:,-1]

    fig = plt.figure(figsize=(8,8))
    fig.set_facecolor('white')
    ax = fig.add_subplot()
    
    ax.plot(final_pred, label='pred') ## 선그래프 생성
    ax.plot(test_true, label='true') 

    ax.legend() ## 범례
    
    plt.title('Forecast power consumption with LSTM', fontsize=20) ## 타이틀 설정
    plt.show()

if __name__ == "__main__":
    main()