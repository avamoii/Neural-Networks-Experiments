import numpy as np
def sigmiod(z):
    """
    Compute the sigmoid activation function.
    
    Arguments:
    z -- A scalar or numpy array of any size.
    
    Return:
    s -- sigmoid(z)
    """
    return 1 / (1 + np.exp(-z))


def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation step.
    This version is fully vectorized to process all 'm' examples simultaneously.
    
    Arguments:
    w -- weights, a numpy array of size (nx, 1)
    b -- bias, a scalar
    X -- input data matrix of size (nx, m)
    Y -- true "label" vector of size (1, m)
    
    Return:
    grads -- dictionary containing the gradients of the weights and bias
    cost -- negative log-likelihood cost for logistic regression
    """
    m = X.shape[1]
    

    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    
    
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    
  
    dZ = A - Y
    dw = (1 / m) * np.dot(X, dZ.T)
    db = (1 / m) * np.sum(dZ)
    
    cost = np.squeeze(cost)
    
    grads = {"dw": dw, "db": db}
    return grads, cost