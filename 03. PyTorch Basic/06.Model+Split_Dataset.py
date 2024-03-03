import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

# Module Class
class Model(nn.Module):
    def __init__(self):
        """신경망에 사용될 계층을 초기화"""
        super().__init__() # 부모 클래스 초기화하면 서브 클래스인 모델에서 부모 클래스의 속성 사용 가능
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
    
    def forward(self, x): # 모듈 클래스는 Callable Type으로 순간 호출 메서드(__call__)가 순방향 메서드 실행
        """모델이 어떤 구조를 갖게 될지를 정의"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

# Non-linear Regression
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
train_dataloader = DataLoader(train_dataset, batch_size = 128, shuffle = True, drop_last = True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = CustomModel().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr = 0.0001)

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
        print(f"Epoch : {epoch + 1:4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}")


# Model evaluation
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

# Model save
torch.save(
    model,
    "../models/model.pt"
)

torch.save(
    model.state_dict(),
    "../models/model_state_dict.pt"
)

# Split Dataset
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
        print(f"Epoch : {epoch + 1:4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}")

with torch.no_grad():
    model.eval()
    for X, y in validation_dataloader:
        X, y = X.to(device), y.to(device)

        output = model(X)
        print(f"X : {X}")
        print(f"Y : {y}")
        print(f"Outputs : {outputs}")
        print("------------------------------")