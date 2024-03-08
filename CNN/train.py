def train(model, data_loader, optimizer, criterion, device):

    model.train()

    total_loss = []

    for input, label in data_loader:

        input = input.to(device) 
        label = label.to(device)

        optimizer.zero_grad()

        pred = model(input)
        loss = criterion(pred, label)

        loss.backward()
        optimizer.step()

        total_loss.append(loss)
    
    return sum(total_loss) / len(total_loss)

