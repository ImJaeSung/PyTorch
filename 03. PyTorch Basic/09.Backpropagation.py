import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        self.x1 = df.iloc[:, 0].values
        self.x2 = df.iloc[:, 1].values
        self.y = df.iloc[:, 3].values
        self.length = len(df)

    def __getitem__(self, index):
        x = torch.FloatTensor([self.x1[index], self.x2[index]])
        y = torch.FloatTensor([int(self.y[index])])
        return x, y

    def __len__(self):
        return self.length
    

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

        self.layer1[0].weight.data = nn.Parameter(
            torch.Tensor([[0.4352, 0.3545],
                        [0.1951, 0.4835]])
        )

        self.layer1[0].bias.data = nn.Parameter(
            torch.Tensor([-0.1419, 0.0439])
        )

        self.layer2[0].weight.data = nn.Parameter(
            torch.Tensor([[-0.1725, 0.1129]])
        )

        self.layer2[0].bias.data = nn.Parameter(
            torch.Tensor([-0.3043])
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
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
optimizer = optim.SGD(model.parameters(), lr = 1)

for epoch in range(1):
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

    print(f"Weight : {list(model.parameters())}")

