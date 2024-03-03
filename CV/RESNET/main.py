from data_utils import load_CIFAR10
from model import ResNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

trainset, validset = load_CIFAR10()

# train_loader = DataLoader(trainset, batch_size = 128, shuffle = True, num_workers = 2)
# test_loader = DataLoader(validset, batch_size = 128, shuffle = False, num_workers = 2)
train_loader = DataLoader(trainset, batch_size = 128, shuffle = True)
valid_loader = DataLoader(validset, batch_size = 128, shuffle = False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ResNet(n_layers = 50, n_classes = 10)  # 예시로 ResNet-50 사용
# model = nn.DataParallel(model)  # 멀티 GPU 사용
model.to(device)

model

# 옵티마이저 및 학습률 스케줄러 설정
optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 0.0001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [32000, 48000], gamma = 0.1)
criterion = nn.CrossEntropyLoss().to(device)

def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    total = 0

    for batch_idx, (X_train, y_train) in enumerate(train_loader):
        X_train, y_train = X_train.to(device), y_train.to(device)
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total += X_train.size(0)
        print(train_loss)

    average_loss = train_loss / total
    print(f'Epoch: {epoch}, Average training loss: {average_loss:.4f}')

def valid(model, valid_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X_valid, y_valid in valid_loader:
            X_valid, y_valid = X_valid.to(device), y_valid.to(device)
            preds = model(X_valid)
            test_loss += criterion(preds, y_valid).item()
            pred = preds.argmax(dim = 1, keepdim = True)
            correct += pred.eq(y_valid.view_as(pred)).sum().item()

    test_loss /= len(valid_loader.dataset)
    error = 100. * (1 - correct / len(valid_loader.dataset))
    print(f'\nTest set: Average loss: {test_loss:.4f}, Error rate: {error:.2f}%\n')

for epoch in tqdm(range(1, 64)):  # 64k 반복에 해당하는 에폭 수
    train(model, train_loader, optimizer, epoch)
    valid(model, valid_loader)
    scheduler.step()