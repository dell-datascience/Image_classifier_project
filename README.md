![image_classifier](https://miro.medium.com/max/600/1*jcrUUS9Z-x5fEHJ5ZSoLMw.jpeg)

# Image classifier with PyTorch

Title: Image classifier with PyTorch

Description: Code developed as capstone project when I took Udacity's AI Programming with Python Nanodegree program. Code for an image classifier was built and them converted into a command line application

#### 1. Train.py 
Trains a new network on a dataset and saves the model as a checkpoint.

Basic usage:
- `python train.py data_directory`

Set directory to save checkpoints: 
- `python train.py data_dir --save_dir save_directory`

Choose architecture: 
- `python train.py data_dir --arch "vgg13"`

Set hyperparameters: 

- `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`

Use GPU for training: 
- `python train.py data_dir --gpu`


#### 2. Predict.py 
Uses the trained network to predict the class for an input flower image along with the probability of that name.

Basic usage: 

- `python predict.py /path/to/image checkpoint`

Options:
- `Return top KK most likely classes: python predict.py input checkpoint --top_k 3`

Use a mapping of categories to real names: 

- `python predict.py input checkpoint --category_names cat_to_name.json`

Use GPU for inference: 

- `python predict.py input checkpoint --gpu`
