"""
This is a module that contains formula's for Activation functions and Loss functions.
The derivatives of each functions are included
List of Functions : ['Relu', 'dRelu', 'Sigmoid', 'dSigmoid', 'Softmax', 'dSoftmax', 'CrossEntropy', 'dCrossEntropy']
"""

import numpy as np

EPSILON_CROSS_ENTROPY = 1e-15


def Relu(x):
    """
    Relu Funtion
    Usage: Activation Function
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("x's type should be %s. But it's %s"%("numpy ndarray",type(x)))
    return np.maximum(0,x)
    # used np.maximum() instead of max() for incase of vector inputs.

def dRelu(x):
    """
    Derivative of Relu Function
    Usage: Used for ChainRule in back propagation.
    Works only with vectors. The element wise dRelu function is
    def dRelu(x):
        if x > 0:
            return 1
        else:
            reutrn 0
     """
    if not isinstance(x, np.ndarray):
        raise TypeError("x's type should be %s. But it's %s"%("numpy ndarray",type(x)))
    x[x > 0] = 1
    x[x <= 0] = 0
    return x

    # used np.squeeze() to decrease the array dimension. np.array([[1, 2, 3]]) is reduced to np.array([1, 2, 3]) for interation

def Sigmoid(x):
    """
    Sigmoid Function
    Usage: Activation Function for classification problems(logistic regression)
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("x's type should be %s. But it's %s"%("numpy ndarray",type(x)))
    x[x<-500] = -500
    x[x> 500] = 500
    return 1/(1+np.exp(x))

def dSigmoid(x):
    """
    Derivative of Sigmoid Function
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("x's type should be %s. But it's %s"%("numpy ndarray",type(x)))
    return Sigmoid(x)*(1-Sigmoid(x))

def Softmax(x):
    """
    Softmax Function
    Usage: Activation Function
    Returns the percentage of each element from sum of the vector.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("x's type should be %s. But it's %s"%("numpy ndarray",type(x)))
    return np.exp(x)/np.sum(np.exp(x), axis = 1)

def dSoftmax(x):
    """
    Derivative of Softmax Function
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("x's type should be %s. But it's %s"%("numpy ndarray",type(x)))
    return  np.exp(x)*(np.sum(np.exp(x),axis = 1) - np.exp(x))/ np.sum(np.exp(x),axis = 1) ** 2

def CrossEntropy(y_hypothesis,y):
    """
    Cross-Entropy Function
    Usage: Loss function
    Input : Can't be minus
    The Total loss should be the sum of whole vector
    """
    if not isinstance(y_hypothesis, np.ndarray):
        raise TypeError("y_hypothesis's type should be %s. But it's %s"%("numpy ndarray",type(y_hypothesis)))
    elif not isinstance(y, np.ndarray):
        raise TypeError("y's type should be %s. But it's %s"%("numpy ndarray",type(y)))


    clippedYh = np.clip(y_hypothesis, EPSILON_CROSS_ENTROPY, 1.0-EPSILON_CROSS_ENTROPY)

    return (-1/y.shape[0]) * (y*np.log(clippedYh) + (1-y)*np.log(clippedYh))

def dCrossEntropy(y_hypothesis, y):
    """
    Derivative Cross-Entropy Funtion
    Usage: For back propagation
    """
    if not isinstance(y_hypothesis, np.ndarray):
        raise TypeError("y_hypothesis's type should be %s. But it's %s"%("numpy ndarray",type(y_hypothesis)))
    elif not isinstance(y, np.ndarray):
        raise TypeError("y's type should be %s. But it's %s"%("numpy ndarray",type(y)))

    y_hypothesis[y_hypothesis <= EPSILON_CROSS_ENTROPY] = EPSILON_CROSS_ENTROPY
    y_hypothesis[1 - y_hypothesis <= EPSILON_CROSS_ENTROPY] = 1 - EPSILON_CROSS_ENTROPY
    return (-1/y.shape[0]) * (y * 1/y_hypothesis - (1-y) * 1/(1-y_hypothesis))