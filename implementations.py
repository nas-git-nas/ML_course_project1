import numpy as np

def mse(y, tx, w):
    """Calculate the loss using either MSE or MAE.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D,). The vector of model parameters.
    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx@w # error of prediction
    return 1/(2*tx.shape[0]) * np.sum(e*e)

def least_squares_gradient(y, tx, w):
    """Computes the gradient at w.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(D, ). The vector of model parameters.
    Returns:
        An array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    return -(1/tx.shape[0]) * tx.T@(y-tx@w)


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.
    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    Returns:
        w: weight after max_iter gradient descent optimization steps
        loss: loss after max_iter gradient descent optimization steps
    """
    w = initial_w
    for _ in range(max_iters):
        dL = least_squares_gradient(y, tx, w)
        w = w - gamma*dL       

    return w, mse(y, tx, w)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (SGD) algorithm.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D,). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    Returns:
        w: weight after max_iter stochastic gradient descent optimization steps
        loss: loss after max_iter stochastic gradient descent optimization steps
    """
    w = initial_w
    for _ in range(max_iters):
        n = int(np.random.rand(1)*tx.shape[0]) # sample random data point, mini-batch size = 1
        dLn = least_squares_gradient(y[n], tx[n,:].reshape(1,tx.shape[1]), w)
        w = w - gamma*dLn       

    return w, mse(y, tx, w)

def least_squares(y, tx):
    """Calculate the least squares solution.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.   
    Returns:
        w_opt: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: mean squared error, scalar.
    """
    w_opt = np.linalg.solve(a=tx.T@tx, b=tx.T@y)

    return w_opt, mse(y, tx, w_opt)

def ridge_regression(y, tx, lambda_):
    """Calculate the ridge regression.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.   
    Returns:
        w_opt: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: mean squared error, scalar.
    """
    w_opt = np.linalg.solve(a=(tx.T@tx + 2*tx.shape[0]*lambda_*np.identity(tx.shape[1])), b=tx.T@y)

    return w_opt, mse(y, tx, w_opt)
