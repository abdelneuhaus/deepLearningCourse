import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

### DATASET ###
### Dataset 1 ###
x = torch.Tensor([[1,0,1,0],[1,0,1,1],[0,1,0,1], [1,1,1,1], [0,0,0,0], [1,1,1,0], [0,1,1,1], 
                [0,0,0,1],[0,1,0,0]])
y = torch.Tensor([[1],[1],[0],[1],[0],[1],[1],[0],[0]])


### INITIALIZATION OF PARAMETERS AND HYPERPARAMETERS ###
numberEpoch = 400
learningRate = 1e-2
inputDim = x.shape[1]
hiddenDim = x.shape[0] 
outputDim = 1


w1 = torch.randn((inputDim, hiddenDim))
b1 = torch.zeros((1, hiddenDim))
w2 = torch.randn((hiddenDim, outputDim))
b2 = torch.zeros((1, outputDim))


### IMPLEMENTING FORWARD AND BACK PROP ###
def sigmoid(z):
    return 1 / (1 + torch.exp(-z))


def forwardPropagation(x, w1, w2, b1, b2):
    tanh = nn.Tanh()
    z1 = torch.mm(x, w1) + b1
    a1 = tanh(z1)
    z2 = torch.mm(a1, w2) + b2
    a2 = sigmoid(z2)
    return a2, a1


def backwardPropagation(x, w1, w2, b1, b2, y):
    m = x.shape[0]
    a2, a1 = forwardPropagation(x, w1, w2, b1, b2)

    dZ2 = a2 - y
    dW2 = (torch.mm(torch.transpose(a1, 0, 1), dZ2)) / m
    dZ2 = dZ2.numpy()
    db2 = np.sum(dZ2, axis = 1, keepdims = True) * (1 / m)
    dZ2 = torch.from_numpy(dZ2)
    db2 = torch.from_numpy(db2)

    dZ1 = torch.mul(torch.mm(torch.transpose(w2, 0, 1), dZ2), (1 - torch.pow(a1, 2)))
    dW1 = torch.mm(torch.transpose(x, 0, 1), dZ1) * (1 / m)
    dZ1 = dZ1.numpy()
    db1 = np.sum(dZ1, axis = 1, keepdims = True) * (1 / m)
    dZ1 = torch.from_numpy(dZ1)
    db1 = torch.from_numpy(db1)

    return dW1, db1, dW2, db2



def updateParameters(learningRate, w1, w2, b1, b2, x, y):
    dW1, db1, dW2, db2 = backwardPropagation(x, w1, w2, b1, b2, y)
    
    w1 = w1 - torch.mul(learningRate, dW1)
    b1 = b1 - torch.mul(learningRate, db1)
    w2 = w2 - torch.mul(learningRate, dW2)
    b2 = b2 - torch.mul(learningRate, db2)

    return w1, b1, w2, b2



def costFunction(y, a2):
    m = y.shape[0]
    loss = y * torch.log(a2) + ((1-y) * torch.log(1-a2))
    cost = - torch.mean(loss)
    return cost



def modelNN(x, y, inputDim, hiddenDim, outputDim, numberEpoch, learningRate, w1, b1, w2, b2):
    for i in range(0, numberEpoch):
        
        a2, a1 = forwardPropagation(x, w1, w2, b1, b2)
        cost = costFunction(y, a2)
        gradients = backwardPropagation(x, w1, w2, b1, b2, y)
        w1, b1, w2, b2 = updateParameters(learningRate, w1, w2, b1, b2, x, y)
        print("Cost function at", i, "iteration :", cost)


def predictions(w1, w2, b1, b2, x, y):
    a2, a1 = forwardPropagation(x, w1, w2, b1, b2)
    a2 = a2.numpy()
    a2 = np.round(a2)
    a2 = torch.from_numpy(a2)
    
    return a2



neuralNetwork = modelNN(x, y, inputDim, hiddenDim, outputDim, numberEpoch, learningRate, w1, b1, w2, b2)
print("")
print('actual :\n', y, '\n')
preds = predictions(w1, w2, b1, b2, x, y)
print('predicted :\n', preds, '\n')

y = y.numpy()
preds = preds.numpy()

accuracy = 0
for i in range (0,len(y)):
    if (y[i] == preds[i]):
        accuracy += 1
accuracy = round(accuracy/len(y), 2)
print('accuracy :', accuracy, '%\n')