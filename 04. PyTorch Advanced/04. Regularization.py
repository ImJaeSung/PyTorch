import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        self.x = df.iloc[:, 0].values
        self.y = df.iloc[:, 1].values
        self.length = len(df)
    
    def __getitem__(self, index):
        x = torch.FloatTensor([self.x[index]**2, self.x[index]]) # W1*X^2 + W2*X 형태
        y = torch.FloatTensor([self.y[index]])
        return x, y
    
    def __len__(self):
        return self.length
    
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)
    
    def forward(self, x):
        x = self.layer(x)
        return x
    
file_path = "../archive/datasets/non_linear.csv"
train_dataset = CustomDataset(file_path)
train_dataloader = DataLoader(train_dataset, batch_size = 16, shuffle = True, drop_last = True)

model = CustomModel().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr = 0.00001)

# L1 regularization (Lasso regularization) : sparse solution
for epoch in range(10000):
    cost = 0.0

    for X, y in train_dataloader:
        X = X.to(device)
        y = y.to(device)

        output = model(X)

        _lambda = 0.5
        l1_loss = sum(p.abs().sum() for p in model.parameters()) # lasso regularization

        loss = criterion(output, y) + _lambda*l1_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss
    
    cost = cost/len(train_dataloader)

    if (epoch + 1)%1000 == 0:
        print(f"Epoch : {epoch + 1:4d}, Cost :{cost:.3f}")

# L2 regularization (Ridge regularization) 
for epoch in range(10000):
    cost = 0.0

    for X, y in train_dataloader:
        X = X.to(device)
        y = y.to(device)

        output = model(X)

        _lambda = 0.5
        l2_loss = sum(p.pow(2.0).sum() for p in model.parameters()) # ridge regularization

        loss = criterion(output, y) + _lambda*l2_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss
    
    cost = cost/len(train_dataloader)

    if (epoch + 1)%1000 == 0:
        print(f"Epoch : {epoch + 1:4d}, Cost :{cost:.3f}")


# Weight decay
model = CustomModel().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr = 0.00001, weight_decay = 0.01) # ridge와 동일

for epoch in range(10000):
    cost = 0.0

    for X, y in train_dataloader:
        X = X.to(device)
        y = y.to(device)

        output = model(X)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss
    
    cost = cost/len(train_dataloader)

    if (epoch + 1)%1000 == 0:
        print(f"Epoch : {epoch + 1:4d}, Cost :{cost:.3f}")

# momentum
model = CustomModel().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr = 0.00001, momentum = 0.9)

model = CustomModel().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr = 0.00001, momentum = 0) # equivalent to SGD

for epoch in range(10000):
    cost = 0.0

    for X, y in train_dataloader:
        X = X.to(device)
        y = y.to(device)

        output = model(X)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss
    
    cost = cost/len(train_dataloader)

    if (epoch + 1)%1000 == 0:
        print(f"Epoch : {epoch + 1:4d}, Cost :{cost:.3f}")

# dropout = voting effect(model averaging)
# order : dropout, batch normalization  
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10)
        self.dropout = nn.Dropout(p = 0.5) # bern parameter p
        self.layer2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

# grdient clipping
model = CustomModel().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr = 0.00001)

for epoch in range(10000):
    cost = 0.0

    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)

        output = model(X)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm(parameters = model.parameters(),
                                max_norm = 0.1, # max_norm 을 초과하는 경우 clipping
                                norm_type = 2.0) # backward 이후와 optimization 사이에 수행
        optimizer.step()

    if (epoch + 1)%1000 == 0:
        print(f"Epoch : {epoch + 1:4d}, Cost :{cost:.3f}")


