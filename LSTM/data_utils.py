import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import random 
import string

f = open('dataset/alphabet.txt', 'r')
data = f.read()

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.data_length = len(self.data)
        self.unique_characters = string.printable 
        self.unique_characters_length = len(self.unique_characters)
        
        self.int2char = {i:char for i, char in enumerate(self.unique_characters)}
        self.char2int = {char:i for i, char in enumerate(self.unique_characters)}

        self.encoding_data = self.chars2int(self.data)

    def __len__(self):
        return self.data_length
    
    def get_data(self, bidirectional):
        x = self.encoding_data[:-1] # Remove last character for input
        y = self.encoding_data[1:]  # Remove first character for ouput

        if bidirectional:
            x += x[::-1]
            y += y[::-1]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'    
        x = torch.Tensor(x).to(device)
        y = torch.Tensor(y).to(device)

        return x, y
    
    def get_valid_data(self, sequence_length):
        x = self.encoding_data[:-1]
        y = self.encoding_data[1:]

        idx = random.randint(0, len(x) - sequence_length)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x = torch.Tensor(x[idx : idx + sequence_length]).to(device)
        y = torch.Tensor(y[idx : idx + sequence_length]).to(device)

        return x, y
    
    def chars2int(self, chars):
        return [self.char2int[c] for c in chars]
    
    def int2chars(self, ints):
        return [self.int2char[c] for c in ints]
