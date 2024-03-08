import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import datetime
from sklearn.preprocessing import MinMaxScaler

def load_data():

    df = pd.read_csv('train.csv')
    df = df[df['건물번호'] == 1]

    return df

def fill_missing_with_avg(df, columns):
    for i in range(len(df)):
        if pd.isna(df.loc[i, columns]):
            
            prev_value_sum = df.loc[i-4:i-1, columns].sum()
            next_value_sum = df.loc[i+1:i+4, columns].sum()
            avg_value = (prev_value_sum + next_value_sum) / 8

            df.loc[i, columns] = avg_value
    return df

def sin_transform(values):
    return np.sin(2 * np.pi * values)

def cos_transform(values):
    return np.cos(2 * np.pi * values)

def CDH(df, num_building):
    df_ = df.copy()
    cdhs = np.array([])
    for num in range(1, num_building+1, 1):
        cdh = []
        cdh_df = df_[df_['building_num'] == num_building]
        cdh_temp = cdh_df['temperature'].values # Series로도 돌릴 수 있지만 array로 돌리는게 속도가 훨씬 빠름
        for i in range(len(cdh_temp)):
            if i < 11:
                cdh.append(np.sum(cdh_temp[:(i+1)] - 26))
            else:
                cdh.append(np.sum(cdh_temp[(i-11):(i+1)] - 26))
        
        cdhs = np.concatenate([cdhs, cdh])
    
    return cdhs

def split_data(data):

    train_set, test_set = train_test_split(data, test_size=24*7, shuffle=False)

    return train_set, test_set

def preprocess(train_set):

    train_set = train_set.drop(columns=['num_date_time', '일조(hr)', '일사(MJ/m2)'])
    train_set.columns = ['building_num', 'date', 'temperature', 'precipitation', 'windspeed', 'humidity', 'power_consumption']

    train_set['precipitation'].fillna(0, inplace=True)
    train_set = fill_missing_with_avg(train_set, 'windspeed')
    train_set = fill_missing_with_avg(train_set, 'humidity')

    train_set['date'] = pd.to_datetime(train_set['date'], format='%Y%m%d %H')

    train_set['month'] = train_set.date.dt.month
    train_set['day'] = train_set.date.dt.day
    train_set['weekday'] = train_set.date.dt.weekday
    train_set['hour'] = train_set.date.dt.hour
    train_set['date'] = train_set.date.dt.date

    weekday_periodic = train_set['weekday'] / 6
    hour_periodic = train_set['hour'] / 23

    train_set['sin_weekday'] = sin_transform(weekday_periodic)
    train_set['cos_weekday'] = cos_transform(weekday_periodic)
    train_set['sin_hour'] = sin_transform(hour_periodic)
    train_set['cos_hour'] = cos_transform(hour_periodic)

    month_dummy = pd.get_dummies(train_set['month']).rename(columns={6:'month_6', 7:'month_7', 8:'month_8'})
    train_set = pd.concat([train_set, month_dummy[['month_6', 'month_7']]], axis=1)

    train_set['holiday'] = train_set.apply(lambda x : 0 if x['day']<5 else 1, axis = 1)
    train_set.loc[(train_set.date == datetime.date(2022, 6, 6))&(train_set.date == datetime.date(2022, 8, 15)), 'holiday'] = 1

    train_set['DI'] = 9/5*train_set['temperature'] - 0.55*(1 - train_set['humidity']/100) * (9/5*train_set['humidity'] - 26) + 32

    train_set['CDH'] = CDH(train_set, 1)

    train_set = train_set.drop(columns=['month', 'date'])
    train_set = pd.concat([train_set.iloc[:,0:5], train_set.iloc[:,6:], train_set.iloc[:,5:6]], axis=1)

    train_set, test_set = split_data(train_set)
    train_set, valid_set = split_data(train_set)

    test_true = test_set.iloc[:,-1].reset_index(drop=True)

    valid_set = pd.concat([train_set[-24:], valid_set]).reset_index(drop=True)
    test_set = pd.concat([valid_set[-24:], test_set]).reset_index(drop=True)
    test_set.iloc[24:,-1] = 0

    return train_set, valid_set, test_set, test_true

class WindowDataset(Dataset):
    def __init__(self, data, input_window, output_window, input_size, stride=1):
        
        L = data.shape[0]
        self.seq_len = input_window + output_window
        num_samples = (L - self.seq_len) // stride + 1
        data_tensor = torch.tensor(data)

        X = torch.zeros(num_samples, input_window, input_size)
        y = torch.zeros(num_samples, output_window)

        for i in range(num_samples):
            
            X[i,:] = data_tensor[i*stride : i*stride+input_window]
            y[i,:] = data_tensor[i*stride+input_window : i*stride+self.seq_len, -1]

        self.x = X
        self.y = y
        self.len = len(X)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len

def loader(train_set, valid_set, test_set, batch_size=64):

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_loader, valid_loader, test_loader