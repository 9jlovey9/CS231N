from builtins import range
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
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    for i in range(N):
        y_hat = X[i] @ W                    # (1,D) * (D,C) -> (1,C)
        y_exp = np.exp(y_hat - y_hat.max()) # 防止指数爆炸
        softmax = y_exp / y_exp.sum()       # 计算概率
        loss -= np.log(softmax[y[i]])       # 交叉熵函数
        softmax[y[i]] -= 1                  # 计算中间梯度
        dW += np.outer(X[i], softmax)       # 计算dW梯度
      
    loss = loss / N + reg * np.sum(W**2)    # 平均
    dW = dW / N + 2 * reg * W               # 加上正则化梯度
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0] # 总样本数（图片数量）
    Y_hat = X @ W  # 计算初始得分，(N,D) * (D,C) ->(N,C)

    P = np.exp(Y_hat - Y_hat.max())      # 归一化，使数值稳定（防止指数爆炸）
    P /= P.sum(axis=1, keepdims=True)    # P(N,C)，每一行都是一个样本，计算每一个样本对应每一类的概率值分布

    loss = -np.log(P[range(N), y]).sum() # 计算每一个样本的损失函数（交叉熵函数）
    loss = loss / N + reg * np.sum(W**2) # 平均+正则化

    P[range(N), y] -= 1                  # 计算中间（P层）的梯度
    dW = X.T @ P / N + 2 * reg * W       # 计算dW的梯度

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
