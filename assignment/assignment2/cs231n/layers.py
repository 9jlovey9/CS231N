from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # 先将输入x进行调整矩阵大小至 x(N,D),D=d_1*d_2*...*d_k表示维度
    #out = x * w + b ==> x(N,D) * w(D,M) + b(M,) = out(N,M), 在分类任务中，M一般表示类别数量
    #out = x.reshape(len(x), -1) @ w + b
    
    out = x.reshape(x.shape[0], -1) @ w + b
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # out = x.reshape(x.shape[0], -1) @ w + b   #(N,D)*(D,M)+(M,) = (N,M)
    # 根据assignment-notes的Section1结论，Y=XW，则有
    # dX = dY @ W.T
    # dW = X.T @ dY
    #print(dout.shape,w.T.shape)
    dx = (dout @ w.T).reshape(x.shape)          # (N,M) * (M,D) -> (N,D) -> (N,d_1,d_2,...,d_k)
    dw = x.reshape(x.shape[0], -1).T @ dout     # (D,N) * (N,M) -> (D,M)
    db = dout.sum(axis=0)                       # b这里是利用了广播特性，从上到下的梯度均为1，加起来即可

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # ReLu函数表达式：
    
    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # 若x>0，则导数为1，若x<=0，则梯度为0
    dx = dout * (x > 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # https://cs231n.github.io/linear-classify/#softmax-classifier
    
    N = len(y)                                                      #N表示样本个数
    #print(x.shape)
    s_stable = x - x.max(axis=1, keepdims=True)                     # 为了防止数值爆炸，先减去最大值，使数值都小于0
    s_stable_exp = np.exp(s_stable)                                 # 指数处理 
    p = s_stable_exp / s_stable_exp.sum(axis=1, keepdims=True)      # 分别求每一个样本不同类别的概率(softmax)
    
    loss = -np.log(p[range(N), y]).sum() / N                        # 交叉熵的损失函数：-y_i log p_i，多样本则求和再平均
    
    dx = p                                                          # 对dx的求导，可以看assignment-notes Section2的推导
    dx[range(N), y] -= 1                                             
    dx /= N
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]         # mode: 'train' or 'test'; required
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype)) #在测试时用到的就是滑动平均值
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # review:axis=0表示从上到下，axis=1从左到右
        # 根据公式一步一步来写 
        # x(N,D) gamma(D,) beta(D) 
        
        mu = x.mean(axis=0)              # (D,) 分别计算N个不同样本同一个特征的均值
        var = x.var(axis=0)              # (D,) 分别计算N个不同样本同一个特征的方差
        std = np.sqrt(var + eps)             # (D,) 计算标准差，加上偏置（防止下面分母为0）
        #std = x.std(axis = 0)           # (D,) 利用np.std可以一步到位
        x_hat = (x - mu) / std           # (N,D) 进行正则化
        out = gamma * x_hat + beta       # (N,D) 对输入x进行缩放和平移

        # 在测试阶段，要计算输入x的滑动平均均值和方差
        running_mean = momentum * running_mean + (1 - momentum) * mu # 更新总体均值
        running_var = momentum * running_var + (1 - momentum) * var  # 更新总体方差
        
        cache = (gamma, beta, mu, var, std, x_hat, eps, x)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    # 下面的方法就是参照了上面网页，利用计算图进行求解
    
    gamma, beta, mu, var, std, x_hat, eps, x = cache
    N = x.shape[0]
    
    # 正向传播：
    # mu = x.mean(axis=0)
    # var = x.var(axis=0)
    # std = np.sqrt(var) + eps
    # x_hat = (x - mu) / std
    # out = gamma * x_hat + beta
    
    dbeta = np.sum(dout, axis=0)            # (D,)
    
    dgamma = np.sum(dout * x_hat, axis=0)   # dout(N,D).*dx_hat(N,D)->(N,D)，这里的正向传播是广播机制，要取sum
    dx_hat = dout * gamma
    
    dstd = -np.sum(dx_hat * (x-mu), axis=0) / (std**2)  #(D,) 这里也是要注意，正向传播利用了广播机制，因此要取sum
    dvar = 0.5 * dstd / std                             #(D,)
    
    #dx由两部分组成，参考计算图即可
    dx1 = dx_hat / std + 2 * (x-mu) * dvar / N 
    dmu = -np.sum(dx1, axis=0)
    dx2 = dmu / N 
    dx = dx1 + dx2
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 这里直接对着论文推导的导数公式即可
    gamma, beta, mu, var, std, x_hat, eps, x = cache
    N = x.shape[0]
    
    S = lambda x: x.sum(axis=0)     # 在BN算法中，都是对同一特征进行求均值，因此axis=0
    
    #dbeta,dgamma和上面是一样的
    dbeta = np.sum(dout, axis=0)            # (D,)
    dgamma = np.sum(dout * x_hat, axis=0)
    
    dx = dout * gamma / (N * std)
    dx = N*dx  - S(dx*x_hat)*x_hat - S(dx)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # 根据提示，可以通过对输入进行转置，然后进行BN的forward
    # 这里采用的是详细版（上面的做法容易出错）
    
    mu = np.mean(x, axis=1, keepdims=True)    # 求均值，注意这里的是axis=1，BN的是axis=0
    std = np.std(x, axis=1, keepdims=True)
    x_hat = (x - mu) / (std + eps)
    out = gamma * x_hat + beta

    cache = (gamma, beta, mu, std, x_hat, eps, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # 基本上和BN是一样的，仅是要注意axis在不同情况下的取值
    # 考虑不同维度所表示的物理意义
    
    gamma, beta, mu, std, x_hat, eps, x = cache
    D = x.shape[1]
    
    dgamma = np.sum(dout * x_hat, axis=0)
    
    dbeta = np.sum(dout, axis=0)
    dx_hat = dout * gamma  # (N, D)

    dvar = dx_hat * (x - mu) * -0.5 * (std + eps)**-2 / std
    dvar = np.sum(dvar, axis=1, keepdims=True)  # (D,)
    
    dmu = -dx_hat / (std + eps) + dvar * -2 * (x - mu) / D
    dmu = np.sum(dmu, axis=1, keepdims=True)  # (D,)
    
    dx = dx_hat / (std + eps) + dvar * 2 * (x - mu) / D + dmu / D

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # 根据上面的提示，这里写的是 inverted 版本，就是训练的时候，就要除以p
        # 这样在测试时不要进行特殊处理
        
        mask = (np.random.randn(*x.shape) < p) / p  #创造概率矩阵，决定每一项的抛弃与否，记得除以p
        out = x * mask  # 将输入与drop层相乘，随机舍弃

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # inverted 做法在测试时，不需要进行其他操作
        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # drop层就是由0 1组成的随机drop矩阵，在1的地方梯度为1
        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 从写法来看，其实和FCN写法差不多，先将网络参数进行赋值，然后初始化，然后正向传播
    # 注意输出输出矩阵的维度即可
    
    pad = conv_param['pad']
    stride = conv_param['stride']
    N, C, H, W = x.shape  #
    F, C, HH, WW = w.shape  #
    h_out = (H - HH + 2*pad) // stride + 1
    w_out = (W - WW + 2*pad) // stride + 1
    
    # 不是很理解为什么要这两行
    # assert (H - HH + 2*pad) % stride == 0
    # assert (W - WW + 2*pad) % stride == 0

    padded_x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')  # (N,C,H,W)
    out = np.zeros((N, F, h_out, w_out))
    w_cur = w.reshape((1, F, -1))  # (1, F, C*HH*WW)
    
    # 这里用双重循环来写，虽然效率可能较低，不过方便理解。
    # 三重循环最容易理解，不过二重的精巧一些
    # 参考代码这部分是利用了window函数，就可以取出窗口，不写双重循环
    
    for hi in range(h_out):
      for wi in range(w_out):
        x_cur = padded_x[:, :, hi*stride:hi*stride+HH, wi*stride:wi*stride+WW]  # (N, C, HH, WW)
        x_cur = x_cur.reshape((N, 1, -1))  # (N, 1, C*HH*WW)
        xw_tmp = x_cur * w_cur  # (N,1,C*HH*WW) * (1,F,C*HH,*WW) -> (N, F, C*HH*WW)
        xw_cur = np.sum(xw_tmp, axis=2) + b  # (N, F)
        out[:, :, hi, wi] = xw_cur

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # 反向传播第一步：取出参数，初始化
    x, w, b, conv_param = cache
    dx = np.zeros_like(x)  # (N, C, H, W)
    dw = np.zeros_like(w)  # (F, C, HH, WW)
    db = np.zeros_like(b)  # (F,)
  
    pad = conv_param['pad']
    stride = conv_param['stride']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    h_out = (H - HH + 2*pad) // stride + 1
    w_out = (W - WW + 2*pad) // stride + 1
    
    #依旧是不知道这列是在干啥，感觉上应该是为了排错，但是我觉得=0也没有问题才对
    #assert (H - HH + 2*pad) % stride == 0
    #ssert (W - WW + 2*pad) % stride == 0
    
    padded_x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    dpadded_x = np.zeros_like(padded_x)
    
    # 因为w和x都是高维的，压缩至三维的情况，将表征特征的都压缩在一维
    # 和正向传播时一致
    
    w_cur = w.reshape((1, F, -1))  # (1, F, C*HH*WW)
    
    # 这里在纸上画出三维图进行推导比较清晰
    # 要注意axis=0/1/2的情况
    
    for h in range(h_out):
      for w in range(w_out):
        x_cur = padded_x[:, :, h*stride:h*stride+HH, w*stride:w*stride+WW]  # (N, C, HH, WW)
        x_cur = x_cur.reshape((N, 1, -1))  # (N, 1, C*HH*WW)
        
        # 根据forward来写backward
        # dw += dout[:, :, h*stride:h*stride+HH, w*stride:w*stride+WW] * x_cur
        # xw_tmp = x_cur * w_cur  # (N, F, C*HH*WW)
        # xw_cur = np.sum(xw_tmp, axis=2) + b  # (N, F)
        # out[:, :, h, w] = xw_cur

        dxw_cur = dout[:, :, h, w]  # (N, F)
        
        db += np.sum(dxw_cur, axis=0)   #(F,)
        
        dxw_cur_tmp = np.expand_dims(dxw_cur, 2)  # 扩充第三个维度(N, F)->(N, F, 1)
        dxw_tmp = np.ones((N, F, C*HH*WW)) * dxw_cur_tmp  # (N, F, C*HH*WW)
        
        dx_cur = dxw_tmp * w_cur  # (N, F, C*HH*WW)
        
        dw_cur = dxw_tmp * x_cur  # (N, F, C*HH*WW)
        
        dw += np.sum(dw_cur, axis=0).reshape(dw.shape)
        
        dpadded_x[:, :, h*stride:h*stride+HH, w*stride:w*stride+WW] += np.sum(dx_cur, axis=1).reshape((N, C, HH, WW))
    dx = dpadded_x[:, :, pad:-pad, pad:-pad]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # 和CNN层的传播写法是一样的，首先计算维度的大小的变化
    # 然后双重循环即可
    
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    out = np.zeros((N, C, H_out, W_out))
    
    for h in range(H_out):
      for w in range(W_out):
        out[:, :, h, w] = x[:, :, h*stride:h*stride+pool_height, w*stride:w*stride+pool_width].max(2).max(2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # maxpool层的反向传播的关键是找到最大值的索引
    # 对最大值的位置梯度为1，其他都为0
    
    # 第一步还是赋值
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride
    dx = np.zeros_like(x)  # N, C, H, W
    
    # 因为写的是二重循环，都需要将四维压缩至三维进行操作
    for h in range(H_out):
      for w in range(W_out):
        x_tmp = x[:, :, h*stride:h*stride+pool_height, w*stride:w*stride+pool_width].reshape(N*C, -1)  # (N,C,pH,pW) -> (N*C, pool_height*pool_width)
        x_idx = np.argmax(x_tmp, axis=1)  # (N*C,1) 找到每一行最大值的索引
        dx_tmp = np.zeros((N*C, pool_height*pool_width))    #除了最大值索引的位置梯度为1，其余都是0
        dx_tmp[range(N*C), x_idx] = dout[:, :, h, w].flatten()  # (N*C, pool_height*pool_width)，最大值索引处梯度为1
        dx[:, :, h*stride:h*stride+pool_height, w*stride:w*stride+pool_width] += dx_tmp.reshape((N, C, pool_height, pool_width))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # 这里本质上就是BN，不过将特征作为一个维度，其余的压缩至一个维度即可
    
    N, C, H, W = x.shape
    x = x.transpose(0, 2, 3, 1).reshape(-1, C)                #(N,C,H,W) -> (N,H,W,C) -> (N*H*W,C)
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)       #(N,H,W,C) -> (N,C,H,W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 这里也是进行维度的压缩，然后和BN的是一致的
    
    N, C, H, W = dout.shape
    dout = dout.transpose(0, 2, 3, 1).reshape(-1, C)        #(N,C,H,W) -> (N,H,W,C) -> (N*H*W,C)
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # 这里的代码也是可以根据BN或者LN的代码修改而来
    
    N, C, H, W = x.shape
    x_group = x.reshape(N, G, -1) #(N,C,H,W) -> (N,G,C/G*H*W)
    mu = np.mean(x_group, axis=2, keepdims=True)  # (N, G)
    std = np.std(x_group, axis=2, keepdims=True)
    x_hat = (x_group - mu) / (std + eps)
    x_hat = x_hat.reshape(x.shape)
    out = gamma * x_hat + beta

    cache = (gamma, beta, mu, std, x_hat, eps, x, G)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    gamma, beta, mu, std, x_hat, eps, x, G = cache
    N, C, H, W = x.shape
    dgamma = np.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True)
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dx_hat = dout * gamma  # (N, C, H, W)

    dx_hat = dx_hat.reshape(N, G, -1)
    x = x.reshape(N, G, -1)
    D = x.shape[2]
    dvar = dx_hat * (x - mu) * -0.5 * (std + eps)**-2 / std
    dvar = np.sum(dvar, axis=2, keepdims=True)  # (N, G)
    dmu = -dx_hat / (std + eps) + dvar * -2 * (x - mu) / D
    dmu = np.sum(dmu, axis=2, keepdims=True)  # (N, G)
    dx = dx_hat / (std + eps) + dvar * 2 * (x - mu) / D + dmu / D
    dx = dx.reshape((N, C, H, W))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
