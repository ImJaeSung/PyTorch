#%%
import torch
import torch.nn as nn

#%%;
class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(VanillaRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.RNN = nn.RNN(
            self.input_size, self.hidden_size, self.n_layers, batch_first = True
            )
        self.fc = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x):
        self.batch_size = x.size(axis = 0)
        hidden = self.init_hidden(self.batch_size)
        out, hidden = self.RNN(x, hidden)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)

        return out, hidden
    
    def init_hidden(self, batch_size):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        hidden = torch.zeros(self.n_layers, self.batch_size, self.hidden_size).to(device)
        return hidden
# %%
