import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class Dataset:
    '''학습에 필요한 데이터 샘플을 정제하고 정답을 저장하는 기능'''
    def __init__(self, data, *arg, **kwargs):
        """입력된 데이터의 전처리 과정 수행"""
        self.data = data
    
    def __getitem__(self, index):
        """입력된 인덱스에 해당하는 데이터 샘플을 불러오고 반환"""
        return tuple(data[index] for data in data.tensors)

    def __len__(self):
        """학습에 사용된 전체 데이터셋의 개수"""
        return self.data[0].size(0)


X_train = torch.FloatTensor([
    [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]
])
y_train = torch.FloatTensor([
    [0.1, 1.5], [1, 2.8], [1.9, 4.1], [2.8, 5.4], [3.7, 6.7], [4.6, 8]
])

train_dataset = TensorDataset(X_train, y_train) # TensorDataset는 기본 Dataset 클래스를 상속받아 재정의된 클래스
train_dataloader = DataLoader(train_dataset, batch_size = 2, shuffle = True, drop_last = True)

model = nn.Linear(2, 2, bias = False)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001)

for epoch in range(20000):
    cost = 0.0

    for batch in train_dataloader:
        X, y = batch
        output = model(X)

        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss
    
    cost = cost / len(train_dataloader)

    if (epoch + 1)%1000 == 0:
        print(f"Epoch : {epoch + 1:4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}")
