import torch

def valid(model, data_loader, criterion, device):

    model.eval()

    total_loss = []
    predictions = []

    with torch.no_grad():
        for i, (input, label) in enumerate(data_loader):

            if i == 0:

                input = input.to(device) 
                label = label.to(device)

                pred = model(input)
                loss = criterion(pred, label)

                predictions.append(pred)
                total_loss.append(loss)

            elif i < 25:

                input = input.to(device) 
                label = label.to(device)

                input[:,-i:,-1] = torch.tensor(predictions, device=device)

                pred = model(input)
                loss = criterion(pred, label)

                predictions.append(pred)
                total_loss.append(loss)

            else:

                input = input.to(device) 
                label = label.to(device)

                input[:,-input.shape[1]:,-1] = torch.tensor(predictions[-input.shape[1]:], device=device)

                pred = model(input)
                loss = criterion(pred, label)

                predictions.append(pred)
                total_loss.append(loss)
    
    return sum(total_loss) / len(total_loss)