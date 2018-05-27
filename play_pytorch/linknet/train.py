import argparse
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable

from model import LinkNet
from loss import cross_entropy_loss_2d

epoch = 800
init_lr = 5e-4

model = LinkNet()
weights = None

loader = camvid_data_loader()
data_loader = torch.utils.data.DataLoader(loader, batch_size=16, shuffle=True, num_workers=multiprocessing.cpu_count())

def train(model, loader, weights, epoch, lr_rate):
    model.cuda()
    model.train(True)

    loss_func = loss.cross_entropy_loss_2d(weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)

    for e in range(epoch):
        aggr_loss = 0
        for i, samples in enumerate(loader):
            raws, labels = Variable(samples['raw']), Variable(samples['label'])
            raws, labels = raws.cuda(), labels.cuda()

            optimizer.zero_grad()

            output = model(raws)
            loss = loss_func(output, labels)
            aggr_loss += loss.data[0]

            print('epoch:', e, 'batch:', i+1, 'aggregate loss:', aggr_loss / (i+1), 'loss:', loss.data[0])

            loss.backward()
            optimizer.step()

            iter_num = len(loader)*e + i

    return model

train(model, data_loader, weights, epoch, init_lr)

