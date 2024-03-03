# save : 파이썬의 pickle을 활용해 파이썬 객체 구조를 Binary protocols로 serialize하여 저장
# import : 저장된 객체 파일을 deserialize하여 현재 프로세스의 메모리에 업로드
# .pt .pth 확장자 사용

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

# model import
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)
    
    def forward(self, x):
        x = self.layer(x)
        return x

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = torch.load("../models/model.pt", map_location = device) # CustomModel을 선언하지 않으면 AttributeError 오류가 발생해 모델을 불러올 수 없음\
print(model)

with torch.no_grad():
    model.eval()
    inputs = torch.FloatTensor(
        [
            [1**2, 1],
            [5**2, 5],
            [11**2, 11]
        ]
    ).to(device)
    outputs = model(inputs)
    print(outputs)


# model state save
torch.save(
    model.state_dict(),
    "../models/model_state_dict.pt"
)

print(model.state_dict()) # 순서가 있는 dictionary 형식으로 반환

# model state import
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)
    
    def forward(self, x):
        x = self.layer(x)
        return x

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_state_dict = torch.load("../models/model_state_dict.pt", map_location = device) # 마찬가지로 CustomModel을 선언하지 않으면 AttributeError 오류가 발생해 모델을 불러올 수 없음\
model.load_state_dict(model_state_dict) # load_state_dict 메서드로 모델 상태를 반영

with torch.no_grad():
    model.eval()
    inputs = torch.FloatTensor(
        [
            [1**2, 1],
            [5**2, 5],
            [11**2, 11]
        ]
    ).to(device)
    outputs = model(inputs)
    print(outputs)


# Checkpoints save
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
dataset = CustomDataset(file_path)
dataset_size = len(dataset)
train_size = int(dataset_size*0.8) 
validation_size = int(dataset_size*0.1)  
test_size = dataset_size - train_size - validation_size

train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])
print(f"Training Data Size : {len(train_dataset)}")
print(f"Validation Data Size : {len(validation_dataset)}")
print(f"Test Data Size : {len(test_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size = 16, shuffle = True, drop_last = True)
validation_dataloader = DataLoader(validation_dataset, batch_size = 4, shuffle = True, drop_last = True)
test_dataloader = DataLoader(test_dataset, batch_size = 4, shuffle = True, drop_last = True)

model = CustomModel().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr = 0.0001)

checkpoint = 1

for epoch in range(10000):
    cost = 0.0

    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)

        output = model(X)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        cost += loss

    cost = cost/len(train_dataloader)

    if (epoch + 1)%1000 == 0: 
        torch.save( # epoch, model.state_dict, optimizer.state_dict 필수로 포함
            {"model" : "CustomModel",
            "epoch" : epoch,
            "model_state_dict" : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict(),
            "cost" : cost,
            "description" : f"CustomModel 체크포인트 = {checkpoint}",
            },
            f"../models/checkpoint-{checkpoint}.pt"
        )
        checkpoint += 1

# checkpoint import
checkpoint = torch.load("../models/checkpoint-6.pt")
model.load_state_dict(checkpoint["model_state_dict"]) # load_state_dict 메서드로 모델 상태를 반영
optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) # load_state_dict 메서드로 optimizer 상태를 반영
checkpoint_epoch = checkpoint["epoch"]
checkpoint_description = checkpoint["description"]
print(checkpoint_description)

for epoch in range(checkpoint_epoch + 1, 10000):
    cost = 0.0

    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)

        output = model(X)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        cost += loss

    cost = cost/len(train_dataloader)

    if (epoch + 1)%1000 == 0:
        print(f"Epoch : {epoch + 1:4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}")