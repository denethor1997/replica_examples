import torch
import math
import numpy as np
from sklearn.metrics import confusion_matrix

class ConfusionMatrix:
    def __init__(self, nclasses, classes):
        self.mat = np.zeros((nclasses, nclasses), dtype=np.float)
        self.valids = np.zeros((nclasses), dtype=np.float)
        self.IoU = np.zeros((nclasses), dtype=np.float)
        self.mIoU = 0

        self.nclasses = nclasses
        self.classes = classes
        self.list_classes = list(range(nclasses))

    def update_matrix(self, target, prediction):
        
