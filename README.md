# Deep-Learning-Project---Crowd-Counting

This is a deep learning model for counting how many people in each image in a mall. The images were captured in a fixed camera with fixed angle. This project adopts CNN as the base model.

## Dataset Description
Please go to the link below to download the dataset:
https://www.kaggle.com/datasets/fmena14/crowd-counting

There are 2 datasets used in this project. 

1. A numpy file containing all 2000 image data.
2. A CSV files that contains labels that represent the number of people in each image.

## The whole model can be divided into the following parts:

1. Data importing.
2. Demonstrating some imported dataset.
3. Exploratory Data Analysis (Number of peopme in the dataset).
4. Splitting training and testing data (1600:400) , and conduct data augmentation and resizing image size to (400,400).
5. Building CNN Model.
6. Compiling the model with Huber loss function, using 'mae' and 'mse' as metrics because this is a regression problem. Training the model with 30 epochs.
7. Comparing 'mae, val_mae, mse, val_mse' improvement with each epoch.
8. Demonstrating some testing and training results in dataframes and images.
