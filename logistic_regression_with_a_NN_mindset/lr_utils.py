import numpy as np
import h5py
    
    
def load_dataset():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    trainSetX_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    trainSetY_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    testSetX_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    testSetY_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    trainSetY_orig = trainSetY_orig.reshape((1, trainSetY_orig.shape[0]))
    testSetY_orig = testSetY_orig.reshape((1, testSetY_orig.shape[0]))
    
    return trainSetX_orig, trainSetY_orig, testSetX_orig, testSetY_orig, classes