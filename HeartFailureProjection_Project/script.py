import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns    # use to plot the correlation matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


###############################
#                             #
#   OPENING THE DATASET FILE  #
#                             #
###############################
dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')
print(dataset)



#############################
#                           #
#   ANALYZING THE DATASET   #
#                           #
#############################
# Checking for null values
print(dataset.isnull().sum())   # no features with missing values


# Checking informations on all features 
print(dataset.info())   # we saw that some data have a float64 type and other an int64 type and no null-value


# Checking correlation between features and death rate using a Spearman correlation test 
# because testing between a variable (feature) and its effect on an other one (death event)
correlation = dataset.corr(method = 'spearman')
sns.heatmap(correlation, annot=True)
# We saw that age (+), time (-), serum_sodium (-), ejection_fraction (-) and serum_creatinine (+) are the most correlated variables to the death rate one. 
# Note that (+) means positive correlation (more of it provokes death rate to increase) and (-) means a negative correlation (the opposite)


# Checking of extrem values with a box plot and removal if there are any
sns.boxplot(x=dataset.ejection_fraction)    # comment the heatmap to avoid display bugs
plt.show()
# There is two values for ejection_fraction that we remove because very large and out of the plot
dataset = dataset[dataset['ejection_fraction']<70]  # keeping only values inferior to 70 (the two values are > to 70)
# we do the same with the other features : nothing to remove (changing the x above)


#################################
#                               #
#   SELECTION THE RIGHT MODEL   #
#        TESTING MODELS         #
#                               #
#################################
"""
We have age, time, serum_creatinine, serum_sodium and ejection_fraction to predict death rate. 
So we have 5 features. 
We don't have labeled data and we want to predict a quantity.
Some choice for the model are : logistic regression, SVM, kNN or Random Forest
We are going to test the accuracy of the first 3 ones
"""
# Creating the training set X and Y
X = dataset[['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']]   # selecting all rows (:) and some columns (, [i,j, k...]) and values allows to only keep the values
Y = dataset['DEATH_EVENT']       # selecting only the death event column and keeping the value only
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.26, random_state = 0)


### Testing a logistic regression ###
logReg = LogisticRegression()             # creating the model
logReg.fit(X_train, y_train)            # training the model
predictionsLR = logReg.predict(X_test)    # testing with our data
# Calculing accuracy
accLR = accuracy_score(y_test, predictionsLR)
print(round(accLR*100, 3), "%")
"""
We get a maximum accuracy of 88,462% with LR and a test set percentage of 0,26 regarding the whole dataset
"""


### Testing a SVM ###
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.19, random_state = 0)
svm = SVC()
svm.fit(X_train, y_train)
predictionsSVM = svm.predict(X_test)
# Calculing accuracy
accSVM = accuracy_score(y_test, predictionsSVM)
print(round(accSVM*100, 3), "%")
"""
We get a maximum accuracy of 92,982% with LR and a test set percentage of 0,19 regarding the whole dataset
"""


### Testing a kNN ###
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.21, random_state = 0)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predictionsKNN = knn.predict(X_test)
# Calculing accuracy
accKNN = accuracy_score(y_test, predictionsKNN)
print(round(accKNN*100, 3), "%")
"""
We get a maximum accuracy of 96,825% with LR and a test set percentage of 0,21 regarding the whole dataset and using k=5 neighbors
"""