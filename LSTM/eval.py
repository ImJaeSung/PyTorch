import torch

def eval(model, data_loader, device):

    model.eval()

    predictions = []

    with torch.no_grad():
        for i, (input, _) in enumerate(data_loader):

            if i == 0:

                input = input.to(device) 
                pred = model(input)
                predictions.append(pred)

            elif i < 25:

                input = input.to(device) 
                input[:,-i:,-1] = torch.tensor(predictions, device=device)
                pred = model(input)
                predictions.append(pred)

            else:

                input = input.to(device) 
                input[:,-input.shape[1]:,-1] = torch.tensor(predictions[-input.shape[1]:], device=device)
                pred = model(input)
                predictions.append(pred)

    return predictions