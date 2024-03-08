import torch

def eval(model, data_loader, device):

    model.eval()  

    correct = 0
    total = 0

    with torch.no_grad():
        for input, label in data_loader:
            
            input = input.to(device)
            label = label.to(device)

            pred = model(input)
            _, pred = torch.max(pred, 1)

            total += label.size(0)
            correct += (pred == label).sum().item()

    accuracy = correct / total

    return accuracy