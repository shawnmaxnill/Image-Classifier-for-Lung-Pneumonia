### Deep Learning Image Classifier for Lung Pneumonia

This repo contains all code used for the neural network (PyTorch) used to detect lung pneumonia.  
Everthing is then wrapped into an API app using FastAPI.  
Testing is done on localhost for now.

Training and Testing data were retrieved from Kaggle available here: 
https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset

### Purpose

This acts basic showcase for computer vision deep learning model.  

### Methods implied for model optimization.

- Early Stopping
- Cross Validation
- Weighted Sampling of Data (data imbalance)
- Data Transformation (Grey but in RGB of 3 channels)
- Microsoft's ResNet18 model


### File contents
`notebooks`  
contains all jupyter notebooks created for exploratory analysis of the data we are working with along with sequential breakdown of the model individually in each column.  

`src`  
This is the main file, contains `dataloader` `models` `utils`  

`dataloader`  
Script for loading in dataset  
Script for loading in dataset into `DataLoader()`

`models`  
Two separate models can be found here:  
ResNet18 (includes backbone freezing functions)  
Basic Convolutional Neural Network

`utils`  
Mainly containing helper functions to piece everything together. (not dynamic)  
- Data transformations for different models: `ResNEt18_transforms.py` `basic_CNN_transforms.py`.  
- Kaggle data cleaning `data_cleaning.py`.  
- Crucial functions for models training and testing loop `train.py` `test.py` `early_stopping`.
- Statistical visualizations and annotations `metrics.py` `plotting.py`

`build.py`  
This is the main script for building the model which is currently set to ResNet18 for higher accuracy with lowest loss so far.  
This script generates a weight.pt file along with a visualization plot.   

`model_app.py`  
This script wraps the model and its weights into an API (FastAPI) in with JSON formatting.  
Currently, running formats of this script still lies within localhost. I suppose one needs to create its own .bash with `curl` to run this.
