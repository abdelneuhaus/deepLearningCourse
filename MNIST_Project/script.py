import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.utils.data
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer



###############################
#                             #
#   OPENING THE DATASET FILE  #
#                             #
###############################
df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')

# Checking for missing value before splitting the dataset df1
print(df1.isnull().values.any())    # there is non NaN in the X data

X = df1.iloc[:, 1:].values  # get the pixel value for each image
y = df1.iloc[:, 0].values   # get the label (number drawn) for each image
# Total of examples : 42000



#############################
#                           #
#   ANALYZING THE DATASET   #
#                           #
#############################
# Checking the shape of both matrix
print("X shape :", X.shape)
print("Y shape : ", y.shape)

# Checking the distribution of the label (y)
sns.countplot(x=y)  # we have a good distribution, with a little bit more of 2




####################################
#                                  #
#      PREPROCESSING THE DATA      #
#                                  #
####################################
"""
We have no dimension reduction to do, we have labelled data and we want to predict numeric. 
We can use SVM, Random Forest, Decision Trees, Naive Bayes or Neural Network
Here, we are going to use neural network because we want to privilege accuracy over speed.
"""

# Creating the training set X and Y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Normalization of pixel values to have number between 0 and 1 (substract mean and normalize variance)
scaler = Normalizer().fit(X_train)
normalizedX = scaler.transform(X_train)
normalizedX_test = scaler.transform(X_test)



#################################
#                               #
#      BUILDING OF THE ANN      #
#                               #
#################################
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

