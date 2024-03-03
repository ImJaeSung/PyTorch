from data_utils import load_CIFAR10
from utils import AverageMeter, accuracy, save_checkpoint
from torch.utils.data import DataLoader
from model import VGGnet
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

train, valid = load_CIFAR10()   
train_loader = DataLoader(train, shuffle = True, batch_size = 256)
valid_loader = DataLoader(valid, shuffle = False, batch_size = 256)

config = {'epochs' : 100,
        'lr' : 0.01, 
        'momentum' : 0.9, 
        'weight_decay' : 5e-4}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = VGGnet(16, batch_norm = True).to(device)
model
optimizer = optim.SGD(model.parameters(),
                    lr = config['lr'], 
                    momentum = config['momentum'],
                    weight_decay = config['weight_decay'])
criterion = nn.CrossEntropyLoss().to(device)

#%%
for epoch in tqdm(range(config['epochs'])):    
    
    lr = config['lr']*(0.5**(epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.train()
    for i, (x_train, y_train) in enumerate(train_loader):

        # measure data loading time
        x_train = x_train.to(device)
        y_train = y_train.to(device)

        # compute output
        preds = model(x_train)
        loss = criterion(preds, y_train)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = preds.float()
        loss = loss.float()
        
        # measure accuracy and record loss
        Acc1, Acc5 = accuracy(preds, y_train, topk = (1, 5))
        losses.update(loss.item(), x_train.size(0))
        top1.update(Acc1.item(), x_train.size(0))
        top5.update(Acc5.item(), x_train.size(0))

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), loss = losses, top1 = top1, top5 = top5))

    # switch to evaluate mode
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.eval()
    with torch.no_grad():

        for i, (x_valid, y_valid) in enumerate(valid_loader):
            x_valid = x_valid.to(device)
            y_valid = y_valid.to(device)

            # compute output
            preds = model(x_valid)
            loss = criterion(preds, y_valid)

            preds = preds.float()
            loss = loss.float()

            # measure accuracy and record loss
            Acc1, Acc5 = accuracy(preds, y_valid, topk = (1, 5))
            losses.update(loss.item(), x_valid.size(0))
            top1.update(Acc1.item(), x_valid.size(0))
            top5.update(Acc5.item(), x_valid.size(0))

            if i % 100 == 0:
                print('Test: [{0}/{1}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(valid_loader), loss = losses, top1 = top1, top5 = top5))

        print(' * Acc@1 {top1.avg:.3f}'.format(top1 = top1))