import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def data_generation(n_row,n_col,seed):
    '''
    1. torch.manual_seed로 재현성 확보
    2. n_row*n_col 크기의 X, n_col*1 크기의 가중치, n_row*1 크기의 잔차 random 생성
    3. Y = XW + e (선형회귀식을 통해 Y 생성)    
    '''
    torch.manual_seed(seed)

    X = torch.rand(n_row,n_col)
    W = torch.rand(n_col,1)
    e = torch.rand(n_row,1)

    Y = torch.mm(X,W) + e

    return X,Y

# x,y = data_generation(1000,4,11)

def data_split(x,y,seed):
    '''
    train_test_split을 이용하여
    train, valid, test = 0.8 : 0.1 : 0.1 로 분할
    '''
    X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2,random_state=seed)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X_test,Y_test,test_size=0.5,random_state=seed)
    
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

# x_train, y_train, x_valid, y_valid, x_test, y_test = data_split(x,y,11)

class CustomDataset(Dataset):
    def __init__(self,x,y):
        self.x_data = x
        self.y_data = y
        
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y


# tr_dataset = CustomDataset(x_train, y_train)
# va_dataset = CustomDataset(x_valid, y_valid)
# te_dataset = CustomDataset(x_test, y_test)

# tr_dataloader = DataLoader(tr_dataset, batch_size=32, shuffle=True)

'''
for x,y in tr_dataloader:
    print(x.shape)
    print(y.shape)
    break
'''
