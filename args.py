import argparse


#args is for training
def args():
    # Create Parse using ArgumentParser
    parser=argparse.ArgumentParser(description='Take user inputs ')
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--data_dir', type = str, default = 'flowers', help = 'Path to the folder of pet images')
    parser.add_argument('--save_dir', type = str, default = './Checkpoint.pth', help = 'Directory to save checkpoint')
    parser.add_argument('--arch', type = str, default = 'vgg13', help = 'CNN Model Architecture [resnet18,alexnet,vgg13]')
    parser.add_argument('--epochs', type = int, default = 10, help = 'Number of epochs')
    parser.add_argument('--learning_rate', type = float, default = 0.01, help = 'Learning rate of model')
    parser.add_argument('--hidden_units', type = int, default = 4, help = 'Number of hidden units')
    parser.add_argument('--print_every', type = int, default = 5, help = 'Print train stats every n epoch')
    parser.add_argument('--gpu', type = str, default = 'cuda', help = 'Use gpu for training and inference')
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()

#pargs is for prediction
def pargs():
    # Create Parse using ArgumentParser
    parser=argparse.ArgumentParser(description='Take user inputs ')
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--data_dir', type = str, default = 'flowers', help = 'Path to the folder of pet images')
    parser.add_argument('--save_dir', type = str, default = './Checkpoint.pth', help = 'Directory to save checkpoint')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'Mapping from category label to category name')
    parser.add_argument('--top_k', type = int, default = 3, help = 'top- K most probable classes')
    parser.add_argument('--input', type = str, default = './flowers/test/7/', help = 'Input figure')
    parser.add_argument('--print_every', type = int, default = 5, help = 'Print train stats every n epoch')
    parser.add_argument('--gpu', type = str, default = 'cuda', help = 'Use gpu for training and inference')
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()



if __name__=='__main__': 
    #validate 
    print(args())