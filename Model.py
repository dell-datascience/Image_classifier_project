from torch import nn
from torchvision import models
from collections import OrderedDict
from args import args

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg13 = models.vgg13(pretrained=True)

models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg13': vgg13}
name=args().arch

def Model():
    model = models[name]
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    #define the feed-forward network with relu with dropout,softmax
#     network=nn.Sequential(OrderedDict([
#                           ('fc1', nn.Linear( 1024,512)),
#                           ('drop', nn.Dropout(p=0.5)),
#                           ('relu', nn.ReLU()),
#                            ('fc2', nn.Linear( 512,256)),
#                           ('drop2', nn.Dropout(p=0.5)),
#                           ('relu2', nn.ReLU()),
#                           ('fc3', nn.Linear(256, 102)),
#                           ('output', nn.LogSoftmax(dim=1))
#                           ]))
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
    print(f"{Model()}\n***Success****")