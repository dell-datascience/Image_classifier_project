
import torch
from torch import nn
from torch import optim
import time
from workspace_utils import active_session
from workspace_utils import active_session
from Model import Model
from Data import load
from args import args

image_datasets, dataloaders=load()

def Train_skynet():  
    trainloader=dataloaders[0]
    testloader=dataloaders[1]
    train_data=image_datasets[0]
    validloader=dataloaders[2]
    print('\nInitiating training sequence\n')
    model=Model()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args().learning_rate)
    device = args().gpu
#     torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Model()
    model.to(device)
    with active_session():
        start = time.time()
        epochs = args().epochs
        steps = 0
        running_loss = 0
        print_every=args().print_every
        train_losses, test_losses = [], []
        print('Commencing training process with {} epochs\n'.format(epochs))
        for epoch in range(epochs):
            for images, labels in trainloader:
                steps+=1
                images, labels = images.to(device), labels.to(device)

                
                log_ps = model.forward(images)
                loss = criterion(log_ps, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            if steps % print_every == 0:

                ## TODO: Implement the validation pass on testloader and print out the validation accuracy
                tot_test_loss=0
                accuracy =0
                model.eval()
                with torch.no_grad():

                    for images, labels in testloader:
                        images, labels = images.to(device), labels.to(device)

                        log_ps=model.forward(images)
                        loss=criterion(log_ps,labels)

                        tot_test_loss+=loss.item()

                        ps=torch.exp(log_ps)
                        top_p,top_class=ps.topk(1,dim=1)
                        equals=top_class==labels.view(*top_class.shape)

                        accuracy +=torch.mean(equals.type(torch.FloatTensor)).item()
                        train_loss=running_loss/print_every
                        test_loss=tot_test_loss/len(testloader)

                        train_losses.append(train_loss)
                        test_losses.append(test_loss)

                        print(f"Training Epoch {epoch+1}/{epochs} "
                              f"Train loss: {running_loss/print_every:.3f}.. "
                              f"Test loss: {test_loss/len(testloader):.3f}.. "
                              f"Test accuracy: {accuracy/len(testloader):.3f}")

                running_loss = 0
                model.train()
    
    
    print('\n\nNow validating Skynet with validation datadet\n\n')
    
    with torch.no_grad():
        for epoch in range(epochs):

            test_loss = 0
            accuracy1 = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy1 += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Validation Epoch {epoch+1}/{epochs}.. "
                          f"Test loss: {test_loss/len(validloader):.3f}.. "
                          f"Test accuracy: {accuracy1/len(validloader):.3f}")   

        
    print('\nSuccessful Training of Skynet\nNow saving as Checkpoint\n')          
    model.class_to_idx=train_data.class_to_idx
    checkpoint = {'input_size': 1024,
                  'output_size': 102,
                  'arch': 'vgg13',
                  'learning_rate': 0.003,
                  'batch_size': 64,
                  'epochs': epochs,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint,args().save_dir )
              
    print('Skynet has been successfully saved to Checkpoint')
    time_elapsed = time.time() - start
    print('\nTotal time: {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed % 60)) 
    
    
    
if __name__=='__main__': 
    Train_skynet()    
    