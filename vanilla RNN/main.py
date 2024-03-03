#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from RNN import VanillaRNN

#%%
text = ["hey how are you", "good i am fine", "have a nice day"]
# Unique characters in sentences
chars = set("".join(text))
# Maps integer to character
int2char = dict(enumerate(chars))
# Maps character to integer
char2int = {char:idx for idx, char in int2char.items()}

# Region Padding sentence
large_sentence_length = len(max(text, key = len)) # longest sentence
for i in range(len(text)):
    while len(text[i]) < large_sentence_length:
        # blank padding
        text[i] += " "   

input_sequences = []
target_sequences = []

for i in range(len(text)):
    # Remove last character for input sequence
    input_sequences.append(text[i][:-1])
    # Remove first character for target sequence
    target_sequences.append(text[i][1:])

# Maps characters to int
for i in range(len(text)):
    input_sequences[i] = [char2int[character] for character in input_sequences[i]]
    target_sequences[i] = [char2int[character] for character in target_sequences[i]]

# Length of different characters
unique_characters_size = len(char2int) # 17
# Size of the sequences
sequences_length = large_sentence_length - 1 # 14
# Size of the batch to train the network
batch_size = len(text) # 3

def one_hot_encode(input_sequences, unique_characters_size, sequences_length, batch_size):
    # Creating a multi_dimensional array of zeros with the desired output shape
    features = np.zeros(
        (batch_size, sequences_length, unique_characters_size), dtype = np.float32
    ) # unique_characters_size for One-Hot Encoding

    # Replacing the at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for j in range(sequences_length):
            # Each character in the input sequence is represented by a number.
            # Therefore, at the position of the number, we can set as 1.
            features[i, j, input_sequences[i][j]]= 1
    
    return features

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Input shape = (Batch Size, Sequence Length, One-Hot Encoding Size)
input_sequences = one_hot_encode(
    input_sequences, unique_characters_size, sequences_length, batch_size
)

# Define the model
input_sequences = torch.from_numpy(input_sequences).to(device)
target_sequences = torch.Tensor(target_sequences).to(device)
print(input_sequences.shape) # (3, 14, 17)
print(target_sequences.shape) # (3, 14)

model = VanillaRNN(
    input_size = unique_characters_size,
    hidden_size = 12,
    output_size = unique_characters_size,
    n_layers = 1
)
model.to(device)

n_epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

# Training the model
for epoch in range(1, n_epochs + 1):
    # Clears exisiting gradients from previous epoch
    optimizer.zero_grad()

    output, hidden = model(input_sequences) # (42, 17), (1, 3, 12)
    loss = criterion(output, target_sequences.view(-1).long()) # 1-D integer array

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print("Epoch {}/{}.....".format(epoch, n_epochs), end = " ")
        print("Loss: {: .4f}".format(loss.item()))

# Predicting
def predict(model, character):
    character = np.array([[char2int[c] for c in character]])
    character = one_hot_encode(character, unique_characters_size, character.shape[1], 1)
    character = torch.from_numpy(character).to(device)
    out, hidden = model(character)
    
    prob = F.softmax(out[-1], dim = 0).data # detach
    # Taking the class with the highest probability score from the ouput
    char_idx = torch.max(prob, dim = 0)[1].item() # get index 

    return int2char[char_idx], hidden

def sample(model, out_len, start = "hey"):
    model.eval()
    with torch.no_grad():
        start = start.lower()
        # First off, run through the starting characters
        chars = [ch for ch in start]
        size = out_len - len(chars)
        # Pass in the previous characters and get a new one
        for _ in range(size):
            char, h = predict(model, chars)
            chars.append(char)

        return "".join(chars)
        
output = sample(model, 15, "hey")
print(output)

# %%
