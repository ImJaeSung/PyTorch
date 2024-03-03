import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, bidirectional = False):
        super(LSTM, self).__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.encoder = nn.Embedding(input_size, self.hidden_size)

        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.n_layers, batch_first = True, bidirectional = self.bidirection)

        if bidirectional:
            self.decoder = nn.Linear(2*self.hidden_size, output_size)
        else:
            self.decoder = nn.Linear(self.hidden_size, output_size)
        
    def forward(self, x, pre_hidden_state):
        x = self.encoder(x)
        output, hidden_state = self.lstm(x, pre_hidden_state)

        
input = torch.zeros(1, 35, 10)
print(input.shape)
lstm = nn.LSTM(10, 20, 2, bidirectional = True, batch_first = True)
lstm

output, (hidden_state, cell_state) = lstm(input)

print(cell_state.shape) # 1*1*20
print(hidden_state.shape) # 1*1*20