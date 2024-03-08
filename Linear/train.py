import torch
import torch.nn as nn

def train(model, loader, criterion, optimizer, device):
    model.train()
    
    total_loss = []
    
    for batch in loader:
        x,true_y = batch 
        
        x = x.to(device)
        true_y = true_y.to(device)

        pred = model(x)
        
        loss = criterion(true_y, pred)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        total_loss.append(loss)
        
    return sum(total_loss)/len(total_loss)

