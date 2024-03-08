import torch

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    
    total_loss = []
    
    for batch in loader:
        x, true_y = batch
        
        x = x.to(device)
        true_y = true_y.to(device)
        
        pred = model(x)
        
        loss = criterion(true_y, pred)
        
        total_loss.append(loss)
        
        return sum(total_loss)/len(total_loss) 