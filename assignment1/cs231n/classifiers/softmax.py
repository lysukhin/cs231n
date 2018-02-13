import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    def _softmax(x):
        x -= np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

    for i in xrange(X.shape[0]):
        f_i = X[i, :].dot(W)
        s_i = _softmax(f_i)
        loss -= np.log(s_i[y[i]] + 1e-8)
        for j in xrange(W.shape[1]):
            if j == y[i]:
                dW[:, j] -= X[i, :]
            dW[:, j] += s_i[j] * X[i, :]

    loss /= X.shape[0]
    loss += 0.5 * reg * np.sum(W * W)
    dW /= X.shape[0]
    dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    F = X.dot(W)

    def _softmax_row_wise(X):
        X -= np.matrix(X.max(axis=1)).T
        return np.exp(X) / np.matrix(np.sum(np.exp(X), axis=1)).T

    S = _softmax_row_wise(F)
    loss -= np.mean(np.log(S[np.arange(S.shape[0]), y].T, axis=0)[0] + 1e-8)
    loss += 0.5 * reg * np.sum(W * W)

    S[np.arange(S.shape[0]), y] -= 1
    dW = X.T.dot(S)
    dW /= X.shape[0]
    dW += reg * W

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
