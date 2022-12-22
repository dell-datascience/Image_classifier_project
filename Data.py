import torch
from torchvision import datasets, transforms, models
import random
from args import args
data_dir=args().data_dir

def load():         
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    training_transforms = transforms.Compose([transforms.Resize(300),
                                             transforms.RandomRotation(30),
                                             transforms.RandomResizedCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([ transforms.Resize(300), 
                                                transforms.RandomResizedCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(300),
                                            transforms.RandomResizedCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder

    train_data = datasets.ImageFolder(train_dir,transform=training_transforms)
    valid_data = datasets.ImageFolder(valid_dir,transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir,transform=testing_transforms) 

    # TODO: Using the image datasets and the trainforms, define the dataloaders

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    testloader  = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    image_datasets=[train_data,valid_data,test_data]
    dataloaders=[trainloader,validloader,testloader]
    
    return image_datasets, dataloaders

if __name__=='__main__':
    print(f"{load()}\n ***Success****")
