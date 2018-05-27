import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

def main():
    
    model = models.__dict__['resnet18']
    model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.1, 
                                momentum=0.9,
                                weight_decay=0.0005)

    cudnn.benchmark = True

    rootdir = '/export/ejanlng/data/imagenet'
    traindir = os.path.join(rootdir, 'train')
    valdir = os.path.join(rootdir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizeCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = None

    train_loader = torch.util.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True,
        num_workers=4, pin_memory=True)

    for epoch in range(0, 90):
        
        lr = 0.1 * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    train(train_loader, model, criterion, optimizer, epoch)


def AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()

    model.train()

    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        losses.update(loss.data[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                  epoch, i, len(train_loader), loss=losses))

if __name__ == '__main__':
    main()


