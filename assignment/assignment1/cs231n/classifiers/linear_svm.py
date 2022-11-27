from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0  #损失函数
    for i in range(num_train):
        scores = X[i].dot(W) #(1,D) * (D,C) --> (1,C)
        correct_class_score = scores[y[i]]  #标签类的得分
        for j in range(num_classes):
            if j == y[i]:   #跳过标签类
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                #########################################################
                #                    START OF CHANGE                    #
                #########################################################
                ds = 1 / num_train    #非标签类
                dys = -1 / num_train    #标签类

                ds_dW = X[i]    #ds/dW
                dW[:,j] += ds * ds_dW    #W:(D,C),第j列表示第j类的权重参数
                dW[:,y[i]] += dys * ds_dW

                # 化简后的写法
                #dW[:, j] += X[i]    # update gradient for incorrect label
                #dW[:, y[i]] -= X[i] # update gradient for correct label
                #########################################################
                #                     END OF CHANGE                     #
                #########################################################

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add L2 regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # dW /= num_train   # scale gradient ovr the number of samples(done before)
    dW += 2 * reg * W # append partial derivative of regularization term(for l2 regularization)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # X.shape = (N,D) = (N,3073)
    # W.shape = (D,C) = (3073,10)
    N = X.shape[0]

    scores = X.dot(W)   #(N,D) * (D,C) -> (N,C)
    scores_label = scores[range(N),[y]].T #(1,N)->(N,1)，此处转置是为了下面的广播

    margins = np.maximum(0, scores - scores_label + 1) #(N,C)
    margins[range(scores.shape[0]), y] = 0;    #标签类不进行损失计算

    loss = np.sum(margins) / N + reg*np.sum(W**2)  #(N,C)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # 由链式法则进行一步一步的回推
    # dloss/dW = dloss/dmargins * dmargins/dscores * dscores/dW
    
    # 构造dloss_tmp,dscores
    dmargins = np.ones_like(margins) / N  #(N,C)
    dscores = np.zeros_like(scores) #(N,C)

    dscores[margins>0] = 1 #margins = np.maximum(0, scores - scores_label + 1)
    dscores[range(N), y] = -np.sum(dscores, axis=1) #
    dscores *= dmargins    # *(1/N)

    # dW(D,C) X(N,D) dscores(N,C)  
    dW = X.T.dot(dscores)

    dW += 2 * reg * W #正则化参数梯度
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
