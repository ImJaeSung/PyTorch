# 활성화 : 뉴럴 네트워크의 뉴런의 출력값을 선형에서 비선형으로 변환하여 
# 데이터의 복잡한 패턴을 기반으로 학습하고 결정을 내릴 수 있게 제어
# 입력을 정규화하는 과정

from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch
import pandas as pd
import torch.optim as optim

class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        self.x1 = df.iloc[:, 0].values
        self.x2 = df.iloc[:, 1].values
        self.x3 = df.iloc[:, 2].values
        self.y = df.iloc[:, 3].values
        self.length = len(df)

    def __getitem__(self, index):
        x = torch.FloatTensor([self.x1[index], self.x2[index], self.x3[index]])
        y = torch.FloatTensor([int(self.y[index])])
        return x, y

    def __len__(self):
        return self.length
    
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(3, 1), # 3차원 -> 1차원
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer(x)
        return x

file_path = "../archive/datasets/binary.csv"
dataset = CustomDataset(file_path)
dataset_size = len(dataset)
train_size = int(dataset_size*0.8)
validation_size = int(dataset_size*0.1)
test_size = dataset_size - train_size - validation_size

train_dataset, validation_dataset, test_dataset = random_split(
    dataset, [train_size, validation_size, test_size], torch.manual_seed(4)
)

train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle = True, drop_last = True)
validation_dataloader = DataLoader(validation_dataset, batch_size = 4, shuffle = True, drop_last = True)
test_dataloader = DataLoader(test_dataset, batch_size = 4, shuffle = True, drop_last = True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = CustomModel().to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr = 0.0001)

for epoch in range(10000):
    cost = 0.0

    for x, y in train_dataloader:
        x, y = x.to(device), y.to(device)

        output = model(x)
        loss = criterion(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss

    cost = cost/len(train_dataloader)

    if (epoch + 1)%1000 == 0:
        print(f"Epoch : {epoch + 1 :4d}, Model : {list(model.parameters())}, Cost : {cost :.3f}")

with torch.no_grad():
    model.eval()
    for x, y in validation_dataloader:
        x, y = x.to(device), y.to(device)

        outputs = model(x)

        print(outputs)
        print(outputs >= torch.FloatTensor([0.5]).to(device)) # 모델의 목적에 따라 임계값을 0.5가 아닌 다른 값으로 설정해 분류 진행 가능
        print('--------------------------------------------')