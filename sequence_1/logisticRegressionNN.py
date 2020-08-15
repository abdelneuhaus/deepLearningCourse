"""
Common steps for pre-processing a new dataset are:

    - Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, ...)
    - Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1)
    - "Standardize" the data
"""


import numpy as np      # scientific computing with Python
import matplotlib.pyplot as plt     # plot graph
import h5py     # read H5 file
import scipy        # test our images
from PIL import Image       # test our images
from scipy import ndimage   # test our images
from lr_utils import load_dataset


### LOADING THE DATA ###
# _orig because we are going to preprocess images to obtain the final version without _orig
trainSetX_orig, trainSetY, testSetX_orig, testSetY, classes = load_dataset()


# Example of a picture's information
index = 24
plt.imshow(trainSetX_orig[index])
print ("y = " + str(trainSetY[:, index]) + ", it's a '" + classes[np.squeeze(trainSetY[:, index])].decode("utf-8") +  "' picture.")



### FINDING THE NUMBER OF TRAIN SET, TEST SET AND SIZE OF AN IMAGE ###
m_train = trainSetX_orig.shape[0]
m_test = testSetY.shape[1]
num_px = trainSetX_orig.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("trainSetX shape: " + str(trainSetX_orig.shape))
print ("trainSetY shape: " + str(trainSetY.shape))
print ("testSetX shape: " + str(testSetX_orig.shape))
print ("testSetY shape: " + str(testSetY.shape))
print("")


### RESHAPE THE TRAINING AND TEST EXAMPLES ###
# After this, our training (and test) dataset is a numpy-array where each column represents a flattened image

trainSetX_flatten = trainSetX_orig.reshape(trainSetX_orig.shape[0], -1).T       # T for transpose of the matrix
testSetX_flatten = testSetX_orig.reshape(testSetX_orig.shape[0], -1).T

print ("trainSetX_flatten shape: " + str(trainSetX_flatten.shape))
print ("trainSetY shape: " + str(trainSetY.shape))
print ("testSetX_flatten shape: " + str(testSetX_flatten.shape))
print ("testSetY shape: " + str(testSetY.shape))
print ("sanity check after reshaping: " + str(trainSetX_flatten[0:5,0]))
print("")


### STANDARDIZATION OF OUR DATASET ###
"""
Substract the mean of the whole numpy array from each example, and then divide each example by the standard deviation of the whole numpy array. 
But for picture datasets, it is simpler and more convenient and works almost as well to just divide every row of the dataset by 255 (the maximum value of a pixel channel).
"""
trainSetX = trainSetX_flatten/255.
testSetX = testSetX_flatten/255.




### BUILDING OUR ALGORITHM ###

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.
    Return:
    s -- sigmoid(z)
    """

    s = 1/(1+np.exp(-z))    
    return s

# Test the function
print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))
print("")



### Initializing parameters ###
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1) -- w will be of shape (num_px * num_px * 3,1)
    b -- initialized scalar (corresponds to the bias)
    """

    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))     # allows to check if some conditions are respected to continue
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

# Test the function
dim = 2
w, b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))
print("")



### FORWARD AND BACKWARD PROPAGATION ###
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """
    
    m = X.shape[1]
    cost = 0
    
    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X) + b)                                    # compute activation function
    cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * (np.log(1-A)))      # compute cost function
    
    # BACKWARD PROPAGATION (TO FIND GRADIENT DESCENT)
    dw = (np.dot(X, (A-Y).T)) / m       # order is important to multiply matrix
    db = (np.sum(A-Y)) / m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


# Test the function
w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))
print("")




### OPTIMIZATION OF W AND B BY THE GRADIENT DESCENT ###
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        # Cost and gradient calculation 
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


# Test the function
params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print("")


### PREDICT FUNCTION HAVING GOOD W AND B ###
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0,i] >= 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
        # Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0        # works also
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction


# Test the function
w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print ("predictions = " + str(predict(w, b, X)))
print("")



### MERGE ALL FUNCTIONS INTO A MODEL ###

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


# Test the function
# Run the model
d = model(trainSetX, trainSetY, testSetX, testSetY, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

# Test on the image 5
index = 5
plt.imshow(testSetX[:,index].reshape((num_px, num_px, 3)))
print ("y = " + str(testSetY[0, index]))
# We get y = 0, it's a cat, but in reality it's a butterfly. The model is overfitting


# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


### Show the cat image from the beginning
plt.show()