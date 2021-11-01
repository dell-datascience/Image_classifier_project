import argparse

def get_inputs():
  
    # Create Parse using ArgumentParser
    parser=argparse.ArgumentParser()
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--data_dir', type = str, default = 'flowers', help = 'path to the folder of pet images')
    parser.add_argument('--save_dir', type = str, default = 'Checkpoint.pth', help = 'directory to save checkpoint')
    parser.add_argument('--arch', type = str, default = 'vgg13', help = 'CNN Model Architecture')
    parser.add_argument('--jason', type = str, default = 'cat_to_name.jason', help = 'mapping from category label to category name')
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()

if __name__=='__main__': 
    #validate 
    args=get_inputs()
    print((args))