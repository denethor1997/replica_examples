import torch

from torch.autograd import Variable

class Train(object):
    def __init__(self, model, data_loader, optimizer, criterion, lr, wd, batch_size):
        super(Train, self).__init__()
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr = lr
        self.wd = wd
        self.bs = batch_size
        
        self.iterations = 0

    def forward(self):
        self.model.train(True)

        total_loss = 0

        for batch_idx, (x, yt) in enumerate(self.data_loader):
            x = x.cuda(async=True)
            yt = yt.cuda(async=True)
            input_var = Variable(x)
            target_var = Variable(yt)

            y = self.model(input_var)
            loss = self.criterion(y, target_var)

            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iterations += 1

        return total_loss*self.bs/len(self.data_loader.dataset)
        
