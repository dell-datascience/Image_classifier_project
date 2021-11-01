import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
import numpy as np
from torch.autograd import Variable
from PIL import Image 
import os, random
from collections import OrderedDict
from get_inputs import get_inputs


def Model():
    
    model = models.vgg13(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    #define the feed-forward network with relu with dropout,softmax
    network=nn.Sequential(OrderedDict([ ('fc1', nn.Linear(25088,1024)),
                              ('drop', nn.Dropout(p=0.5)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear( 1024,512)),
                              ('drop', nn.Dropout(p=0.5)),
                              ('relu', nn.ReLU()),
                               ('fc3', nn.Linear( 512,256)),
                              ('drop2', nn.Dropout(p=0.5)),
                              ('relu2', nn.ReLU()),
                              ('fc4', nn.Linear(256, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier =network
    return model


if __name__=='__main__':
#     arch=get_inputs().arch
    print(Model())
#     print(arch)