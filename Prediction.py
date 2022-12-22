
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.autograd import Variable
from torch import nn
import numpy as np
import time
from PIL import Image 
import os, random
from Model import Model
from Data import load
import json
from args import pargs
import torchvision

model=Model()
optimizer = optim.Adam(model.classifier.parameters(), lr=pargs().learning_rate)
with open(pargs().category_names, 'r') as f:
    cat_to_name = json.load(f)
    
device = pargs().gpu
# torch.device("cuda" if torch.cuda.is_available() else "cpu")  

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    learning_rate=checkpoint['learning_rate']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model

model = load_checkpoint(pargs().save_dir)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    im = im.resize((256,256))
    value = 0.5*(256-224)
    im = im.crop((value,value,256-value,256-value))
    im = np.array(im)/255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std

    return im.transpose(2,0,1)


def predict(image_path,model= model, topk=pargs().top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    # turn off dropout
    model.eval()
    # The image
    image = process_image(image_path)
    
    # tranfer to tensor
    image = torch.from_numpy(np.array([image])).float()
    
    # The image becomes the input
    image = Variable(image)
    
    image = image.to(device)
        
    output = model.forward(image)
    
    probabilities = torch.exp(output).data
    
    # getting the topk (=5) probabilites and indexes
    # 0 -> probabilities
    # 1 -> index
    prob = torch.topk(probabilities, topk)[0].tolist()[0] # probabilities
    index = torch.topk(probabilities, topk)[1].tolist()[0] # index
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    # transfer index to label
    label = []
    for i in range(topk):
        label.append(ind[index[i]])

    return prob, label


# TODO: Implement the code to predict the class from an image file

def Run():
    img = random.choice(os.listdir(pargs().input))
    img_path = pargs().input + img
    prob, classes = predict(img_path) 
    print(prob)
    print(classes)
    print([cat_to_name[x] for x in classes])
    max_index = np.argmax(prob)
    max_probability = prob[max_index]
    label = classes[max_index]

    fig = plt.figure(figsize=(6,6))
    ax1 = plt.subplot2grid((15,9), (0,0), colspan=9, rowspan=9)
    ax2 = plt.subplot2grid((15,9), (9,2), colspan=5, rowspan=5)

    image = Image.open(img_path)
    ax1.axis('off')
    ax1.set_title(cat_to_name[label])
    ax1.imshow(image)

    labels = []
    for cl in classes:
        labels.append(cat_to_name[cl])

    y_pos = np.arange(5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel('Probability')
    ax2.invert_yaxis()
    ax2.barh(y_pos, prob, xerr=0, align='center', color='blue')

    plt.show()
    
    return None
    
if __name__ == '__main__':
    Run()
    
