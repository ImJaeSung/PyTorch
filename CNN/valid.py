import torch

def valid(model, data_loader, criterion, device):

    model.eval()

    total_loss = []

    with torch.no_grad():
        for input, label in data_loader:

            input = input.to(device) 
            label = label.to(device)

            pred = model(input)
            loss = criterion(pred, label)

            total_loss.append(loss)
    
    return sum(total_loss) / len(total_loss)