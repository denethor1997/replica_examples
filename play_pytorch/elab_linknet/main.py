import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from train import Train

from model import LinkNet
from torchvision.models import resnet18

def main():
    cudnn.benchmark = True
    torch.set_default_tensor_type('torch.FloatTensor')

    n_classes = 11
    epoch = 0
    lr = 5e-4
    wd = ''
    bs = 16
    maxepoch = 30

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    criterion_weight = 1/np.log(1.02 + hist)
    criterion_weight[0] = 0
    criterion = nn.NLLLoss(Variable(torch.from_numpy(criterion_weight).float().cuda()))
    print('{}Using weighted criterion{}!!!'.format(CP_Y, CP_C))

    train = Train(model, data_loader_train, optimizer, criterion, lr, wd)

    while epoch <= maxepoch:
        train_error = train.forward()
        
        epoch += 1

if __name__ == '__main__':
    main()
     
