# Computer Vision

This is a note on the course of Stanford CS231n in the year 2022.

2022 Course Website: [Stanford University CS231n: Deep Learning for Computer Vision](http://cs231n.stanford.edu/schedule.html)

| Topic                                                | Chapter |
| ---------------------------------------------------- | ------- |
| Deep Learning Basics                                 | 2 - 4   |
| Perceiving and Understanding the Visual World        | 5 - 12  |
| Reconstructing and Interacting with the Visual World | 13 - 16 |
| Human-Centered Applications and Implications         | 17 - 18 |

## 1 - Introduction

A brief history of computer vision & deep learning...

## 2 - Image Classification

**Image Classification:** A core task in Computer Vision. The main drive to the progress of CV.

**Challenges:** Viewpoint variation, background clutter, illumination, occlusion, deformation, intra-class variation...

**Distance Metric** to campare images:

$$
\text{L1(Manhattan)distances: }d_1(I_1,I_2)=\sum_P |I_1^P-I_2^P|
$$

$$
\text{L2(Euclidean)distance: }d_2(I_1,I_2)=\sqrt{\sum_P (I_1^P-I_2^P)^2}
$$

![](D:\OneDrive%20-%20中山大学\研0\Stanford-CS231n-2021-and-2022\assets\2022-11-18-16-26-28-image.png)

```python
# NN-classisfier

import numpy as np

class NearestNeighbor:
    def _init_(self):
        pass

    # Memorize training data
    def train(self, X, y):
        """ X is N x D where each row is an example.Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    # For each test image:
    # Find closest trin image
    # Predict label of nearest image

    def predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        # loop over all test rows
        for i in xrange(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            min_index = np.argmin(distances) # get the index with smallest distance
            Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

        return Ypred
```

### K Nearest Neighbor

**Hyperparameters:** Distance metric ($p$ norm), $k$ number.

Choose hyperparameters using validation set.

<mark>Never use k-Nearest Neighbor with pixel distance.(?)</mark>

![](D:\OneDrive%20-%20中山大学\研0\Stanford-CS231n-2021-and-2022\assets\2022-11-18-16-25-20-image.png)

### 

![](D:\OneDrive%20-%20中山大学\研0\Stanford-CS231n-2021-and-2022\assets\2022-11-18-16-34-07-image.png)

### Linear Classifier

![](D:\OneDrive%20-%20中山大学\研0\Stanford-CS231n-2021-and-2022\assets\2022-11-18-16-34-53-image.png)

线性代数理解：

![](D:\OneDrive%20-%20中山大学\研0\Stanford-CS231n-2021-and-2022\assets\2022-11-18-16-36-36-image.png)

模板匹配理解：

![](D:\OneDrive%20-%20中山大学\研0\Stanford-CS231n-2021-and-2022\assets\2022-11-18-16-37-55-image.png)

空间几何学理解：

![](D:\OneDrive%20-%20中山大学\研0\Stanford-CS231n-2021-and-2022\assets\2022-11-18-16-38-24-image.png)

线性分类器有着许多的局限性：

![](D:\OneDrive%20-%20中山大学\研0\Stanford-CS231n-2021-and-2022\assets\2022-11-18-16-39-41-image.png)

## 3 - Loss Functions and Optimization

### 3.1 Loss Functions

| Dataset                           | $\big\{(x_i,y_i)\big\}_{i=1}^N\\$                                   |
| --------------------------------- | ------------------------------------------------------------------- |
| Loss Function                     | $L=\frac{1}{N}\sum_{i=1}^NL_i\big(f(x_i,W),y_i\big)\\$              |
| Loss Function with Regularization | $L=\frac{1}{N}\sum_{i=1}^NL_i\big(f(x_i,W),y_i\big)+\lambda R(W)\\$ |

#### 3.1.1 SVM loss

**the SVM loss**: $s=f(x_i,W)$，$s_{yi}$ means the correct label score

$$
\begin{aligned} 
L_{i} &=\sum_{j \neq y_{i}}\left\{\begin{array}{ll}0 & \text { if } s_{y_{i}} \geq s_{j}+1 \\ s_{j}-s_{y_{i}}+1 & \text { otherwise }\end{array}\right.\\ &=\sum_{j \neq y_{i}} \max \left(0, s_{j}-s_{y_{i}}+1\right)
 \end{aligned}
$$

<img title="" src="file:///D:/OneDrive%20-%20中山大学/研0/Stanford-CS231n-2021-and-2022/assets/2022-11-18-17-10-25-image.png" alt="" width="303" data-align="center">

if $L_i=\sum_{j\ne y_1} \max(0,s_j-s_{y_1}+1)^2$

<img title="" src="file:///D:/OneDrive%20-%20中山大学/研0/Stanford-CS231n-2021-and-2022/assets/2022-11-18-17-17-35-image.png" alt="" width="272" data-align="center">

```python
def L_i_vectorized(x, y , W):
    scores = W.dot(x)
    margins = np.maximum(0, scores - socres[y] + 1)
    loss_i = np.sum(margins)
    return loss_i
```

#### Regularization

<img title="" src="file:///D:/OneDrive%20-%20中山大学/研0/Stanford-CS231n-2021-and-2022/assets/2022-11-18-19-47-53-image.png" alt="" data-align="center" width="505">

![](D:\OneDrive%20-%20中山大学\研0\Stanford-CS231n-2021-and-2022\assets\2022-11-18-19-52-42-image.png)

Why regularize?

- Express preferences over weights

- Make the model *simple* so it works on test data

- Improve optimization by adding curvature

L1 and L2 regularization choice:

- L2 regularization likes to  “spread out” the weights（考虑整体分布，整体分布较小）

- L1 prefers sparse solutions(稀疏解)

#### 3.1.2 Softmax classifier(Multinomial Logisitic Regression)

**Motivation:** Want to interpret raw classifier scores as probabilities.

**scores = unnormailzed log probabilities of the classes.**

<img title="" src="file:///D:/OneDrive%20-%20中山大学/研0/Stanford-CS231n-2021-and-2022/assets/2022-11-18-20-45-49-image.png" alt="" width="471">

$s = f(x_i;W)$   **Softmax Function:**  $P(Y=k|X=x_i)=\frac{e^s k}{\sum_j e^{s_j}}$

<mark>Q:At initialization all s will beapproximately equal;what is the loss?</mark>
A:$-\log(1/C)=log(C)$ ,If $C= 10$,then $L=\log(10)=2.3$

Kullback-Leibler dvergence（KL散度）:  $Q(y)$表示真实概率

$$
D_{KL}(P||Q)=\sum_y P(y) \log \frac{P(y)}{Q(y)}
$$

Cross Entropy 交叉熵：

$$
H(P,Q)=H(p)+D_{KL}(P||Q)
$$

  

| Softmax Classifier                     | $p_i=Softmax(y_i)=\frac{\exp(y_i)}{\sum_{j=1}^N\exp(y_j)}\\$ |
| -------------------------------------- | ------------------------------------------------------------ |
| Cross Entropy Loss                     | $L_i=-y_i\log p_i\\$                                         |
| Cross Entropy Loss with Regularization | $L=-\frac{1}{N}\sum_{i=1}^Ny_i\log p_i+\lambda R(W)\\$       |

#### 3.1.3 Softmax vs. SVM

![](D:\OneDrive%20-%20中山大学\研0\Stanford-CS231n-2021-and-2022\assets\2022-11-20-11-11-12-image.png)

softmax:

$$
L_i=-\log\left(\frac{e^{e_{y_i}}}{\sum_j e^{s_j}} \right)
$$

SVM:

$$
L_i=\sum_{j \ne j_i} \max(0,s_j-s_{y_i} + 1)
$$

Loss function:

$$
L = \frac{1}{N} \sum_{i=1}^N L_i + R(W)
$$

<img src="pics\3-loss.png" style="zoom:80%;" />

### 

### 3.2 Optimization

#### SGD with Momentum

**Problems that SGD can't handle:**

1. Inequality of gradient in different directions.
2. Local minima and saddle point (much more common in high dimension).
3. Noise of gradient from mini-batch.

**Momentum:** Build up “velocity” $v_t$ as a running mean of gradients.

| SGD                               | SGD + Momentum                                                                              |
| --------------------------------- | ------------------------------------------------------------------------------------------- |
| $x_{t+1}=x_t-\alpha\nabla f(x_t)$ | $\begin{aligned}v_{t+1}&=\rho v_t+\nabla f(x_t)\\ x_{t+1}&=x_t-\alpha v_{t+1}\end{aligned}$ |
| Naive gradient descent.           | $\rho$ gives "friction", typically $\rho=0.9,0.99,0.999,...$                                |

**Nesterov Momentum:** Use the derivative on point $x_t+\rho v_t$ as gradient instead point $x_t$.

| Momentum                                                                                   | Nesterov Momentum                                                                                   |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| $\begin{aligned}v_{t+1}&=\rho v_t+\nabla f(x_t)\\x_{t+1}&=x_t-\alpha v_{t+1}\end{aligned}$ | $\begin{aligned}v_{t+1}&=\rho v_t+\nabla f(x_t+\rho v_t)\\x_{t+1}&=x_t-\alpha v_{t+1}\end{aligned}$ |
| Use gradient at current point.                                                             | Look ahead for the gradient in velocity direction.                                                  |

<img src="pics\3-momentum.png" style="zoom:80%;" />

#### AdaGrad and RMSProp

**AdaGrad:** Accumulate squared gradient, and gradually decrease the step size.

**RMSProp:** Accumulate squared gradient while decaying former ones, and gradually decrease the step size. ("Leaky AdaGrad")

| AdaGrad                                                                                                                                                       | RMSProp                                                                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\begin{aligned}\text{Initialize:}&\\&r:=0\\\text{Update:}&\\&r:=r+\Big[\nabla f(x_t)\Big]^2\\&x_{t+1}=x_t-\alpha\frac{\nabla f(x_t)}{\sqrt{r}}\end{aligned}$ | $\begin{aligned}\text{Initialize:}&\\&r:=0\\\text{Update:}&\\&r:=\rho r+(1-\rho)\Big[\nabla f(x_t)\Big]^2\\&x_{t+1}=x_t-\alpha\frac{\nabla f(x_t)}{\sqrt{r}}\end{aligned}$ |
| Continually accumulate squared gradients.                                                                                                                     | $\rho$ gives "decay rate", typically $\rho=0.9,0.99,0.999,...$                                                                                                             |

#### Adam

Sort of like "RMSProp + Momentum".

| Adam (simple version)                                                                                                                                                                                                             | Adam (full version)                                                                                                                                                                                                                                                                                   |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\begin{aligned}\text{Initialize:}&\\&r_1:=0\\&r_2:=0\\\text{Update:}&\\&r_1:=\beta_1r_1+(1-\beta_1)\nabla f(x_t)\\&r_2:=\beta_2r_2+(1-\beta_2)\Big[\nabla f(x_t)\Big]^2\\&x_{t+1}=x_t-\alpha\frac{r_1}{\sqrt{r_2}}\end{aligned}$ | $\begin{aligned}\text{Initialize:}\\&r_1:=0\\&r_2:=0\\\text{For }i\text{:}\\&r_1:=\beta_1r_1+(1-\beta_1)\nabla f(x_t)\\&r_2:=\beta_2r_2+(1-\beta_2)\Big[\nabla f(x_t)\Big]^2\\&r_1'=\frac{r_1}{1-\beta_1^i}\\&r_2'=\frac{r_2}{1-\beta_2^i}\\&x_{t+1}=x_t-\alpha\frac{r_1'}{\sqrt{r_2'}}\end{aligned}$ |
| Build up “velocity” for both gradient and squared gradient.                                                                                                                                                                       | Correct the "bias" that $r_1=r_2=0$ for the first few iterations.                                                                                                                                                                                                                                     |

#### Overview

| <img src="pics\3-optimization_overview.gif" style="zoom:70%;" /> | <img src="pics\3-optimization_overview2.gif" style="zoom:70%;" /> |
|:----------------------------------------------------------------:|:-----------------------------------------------------------------:|

#### Learning Rate Decay

Reduce learning rate at a few fixed points to get a better convergence over time.

$\alpha_0$ : Initial learning rate.

$\alpha_t$ : Learning rate in epoch $t$.

$T$ : Total number of epochs.

| Method       | Equation                                                                                     | Picture                                                         |
| ------------ | -------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Step         | Reduce $\alpha_t$ constantly  in a fixed step.                                               | <img src="pics\3-learning_rate_step.png" style="zoom:30%;" />   |
| Cosine       | $\begin{aligned}\alpha_t=\frac{1}{2}\alpha_0\Bigg[1+\cos(\frac{t\pi}{T})\Bigg]\end{aligned}$ | <img src="pics\3-learning_rate_cosine.png" style="zoom:30%;" /> |
| Linear       | $\begin{aligned}\alpha_t=\alpha_0\Big(1-\frac{t}{T}\Big)\end{aligned}$                       | <img src="pics\3-learning_rate_linear.png" style="zoom:30%;" /> |
| Inverse Sqrt | $\begin{aligned}\alpha_t=\frac{\alpha_0}{\sqrt{t}}\end{aligned}$                             | <img src="pics\3-learning_rate_sqrt.png" style="zoom:30%;" />   |

High initial learning rates can make loss explode, linearly increasing learning rate in the first few iterations can prevent this.

**Learning rate warm up:**

<img src="pics\3-learning_rate_increase.png" style="zoom:60%;" />

**Empirical rule of thumb:** If you increase the batch size by $N$, also scale the initial learning rate by $N$ .

#### Second-Order Optimization

|              | Picture                                                 | Time Complexity                     | Space Complexity                    |
| ------------ | ------------------------------------------------------- | ----------------------------------- | ----------------------------------- |
| First Order  | <img src="pics\3-first_order.png" style="zoom:50%;" />  | $O(n)$                              | $O(n)$                              |
| Second Order | <img src="pics\3-second_order.png" style="zoom:50%;" /> | $O(n^2)$ with **BGFS** optimization | $O(n)$ with **L-BGFS** optimization |

**L-BGFS :** Limited memory BGFS.

1. Works very well in full batch, deterministic $f(x)$.
2. Does not transfer very well to mini-batch setting.

#### Summary

| Method         | Performance                                                                  |
| -------------- | ---------------------------------------------------------------------------- |
| Adam           | Often chosen as default method.<br>Work ok even with constant learning rate. |
| SGD + Momentum | Can outperform Adam.<br>Require more tuning of learning rate and schedule.   |
| L-BGFS         | If can afford to do full batch updates then try out.                         |

## 

## 4 - Neural Networks and Backpropagation

### Neural Networks

**Motivation:** Inducted bias can appear to be high when using human-designed features.

**Activation:** Sigmoid, tanh, ReLU, LeakyReLU...

**Architecture:** Input layer, hidden layer, output layer.

**Do not use the size of a neural network as the regularizer. Use regularization instead!**

**Gradient Calculation:** Computational Graph + Backpropagation.

### Backpropagation

Using Jacobian matrix to calculate the gradient of each node in a computation graph.

Suppose that we have a computation flow like this:

<img src="pics\4-graph.png" style="zoom:30%;" />

| Input X                                               | Input W                                                                                                                                            | Output Y                                              |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| $X=\begin{bmatrix}x_1\\x_2\\\vdots\\x_n\end{bmatrix}$ | $W=\begin{bmatrix}w_{11}&w_{12}&\cdots&w_{1n}\\w_{21}&w_{22}&\cdots&w_{2n}\\\vdots&\vdots&\ddots&\vdots\\w_{m1}&w_{m2}&\cdots&w_{mn}\end{bmatrix}$ | $Y=\begin{bmatrix}y_1\\y_2\\\vdots\\y_m\end{bmatrix}$ |
| $n\times 1$                                           | $m\times n$                                                                                                                                        | $m\times 1$                                           |

After applying feed forward, we can calculate gradients like this:

<img src="pics\4-graph2.png" style="zoom:30%;" />

| Derivative Matrix of X                                                                                                                      | Jacobian Matrix of X                                                                                                                                                                                                                                                                                                                                                                                    | Derivative Matrix of Y                                                                                                                      |
| ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| $D_X=\begin{bmatrix}\frac{\partial L}{\partial x_1}\\\frac{\partial L}{\partial x_2}\\\vdots\\\frac{\partial L}{\partial x_n}\end{bmatrix}$ | $J_X=\begin{bmatrix}\frac{\partial y_1}{\partial x_1}&\frac{\partial y_1}{\partial x_2}&\cdots&\frac{\partial y_1}{\partial x_n}\\\frac{\partial y_2}{\partial x_1}&\frac{\partial y_2}{\partial x_2}&\cdots&\frac{\partial y_2}{\partial x_n}\\\vdots&\vdots&\ddots&\vdots\\\frac{\partial y_m}{\partial x_1}&\frac{\partial y_m}{\partial x_2}&\cdots&\frac{\partial y_m}{\partial x_n}\end{bmatrix}$ | $D_Y=\begin{bmatrix}\frac{\partial L}{\partial y_1}\\\frac{\partial L}{\partial y_2}\\\vdots\\\frac{\partial L}{\partial y_m}\end{bmatrix}$ |
| $n\times 1$                                                                                                                                 | $m\times n$                                                                                                                                                                                                                                                                                                                                                                                             | $m\times 1$                                                                                                                                 |

| Derivative Matrix of W                                                                                                                                                                                                                                                                                                                                                                                         | Jacobian Matrix of W                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Derivative Matrix of Y                                                                                                                      |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| $W=\begin{bmatrix}\frac{\partial L}{\partial w_{11}}&\frac{\partial L}{\partial w_{12}}&\cdots&\frac{\partial L}{\partial w_{1n}}\\\frac{\partial L}{\partial w_{21}}&\frac{\partial L}{\partial w_{22}}&\cdots&\frac{\partial L}{\partial w_{2n}}\\\vdots&\vdots&\ddots&\vdots\\\frac{\partial L}{\partial w_{m1}}&\frac{\partial L}{\partial w_{m2}}&\cdots&\frac{\partial L}{\partial w_{mn}}\end{bmatrix}$ | $J_W^{(k)}=\begin{bmatrix}\frac{\partial y_k}{\partial w_{11}}&\frac{\partial y_k}{\partial w_{12}}&\cdots&\frac{\partial y_k}{\partial w_{1n}}\\\frac{\partial y_k}{\partial w_{21}}&\frac{\partial y_k}{\partial w_{22}}&\cdots&\frac{\partial y_k}{\partial w_{2n}}\\\vdots&\vdots&\ddots&\vdots\\\frac{\partial y_k}{\partial w_{m1}}&\frac{\partial y_k}{\partial w_{m2}}&\cdots&\frac{\partial y_k}{\partial w_{mn}}\end{bmatrix}$<br>$J_W=\begin{bmatrix}J_W^{(1)}&J_W^{(2)}&\cdots&J_W^{(m)}\end{bmatrix}$ | $D_Y=\begin{bmatrix}\frac{\partial L}{\partial y_1}\\\frac{\partial L}{\partial y_2}\\\vdots\\\frac{\partial L}{\partial y_m}\end{bmatrix}$ |
| $m\times n$                                                                                                                                                                                                                                                                                                                                                                                                    | $m\times m\times n$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | $ m\times 1$                                                                                                                                |

For each element in $D_X$ , we have:

$D_{Xi}=\frac{\partial L}{\partial x_i}=\sum_{j=1}^m\frac{\partial L}{\partial y_j}\frac{\partial y_j}{\partial x_i}\\$

## 

## 5 - Convolutional Neural Networks

### Convolution Layer

#### Introduction

**Convolve a filter with an image:** Slide the filter spatially within the image, computing dot products in each region.

Giving a $32\times32\times3$  image and a $5\times5\times3$ filter, a convolution looks like:

<img src="pics\5-convolution.png" style="zoom:50%;" />

Convolve six $5\times5\times3$ filters to a $32\times32\times3$ image with step size $1$, we can get a $28\times28\times6$ feature:

<img src="pics\5-convolution_six_filters.png" style="zoom:60%;"/>

With an activation function after each convolution layer, we can build the ConvNet with a sequence of convolution layers:

<img src="pics\5-convolution_net.png" style="zoom:60%;"/>

By **changing the step size** between each move for filters, or **adding zero-padding** around the image, we can modify the size of the output:

<img src="pics\5-convolution_padding.png" style="zoom:60%;"/>

#### $1\times1$ Convolution Layer

This kind of layer makes perfect sense. It is usually used to change the dimension (channel) of features.

A $1\times1$ convolution layer can also be treated as a full-connected linear layer.

<img src="pics\5-convolution_1times1.png" style="zoom:60%;"/>

#### Summary

| **Input**                 |                              |
| ------------------------- | ---------------------------- |
| image size                | $W_1\times H_1\times C$      |
| filter size               | $F\times F\times C$          |
| filter number             | $K$                          |
| stride                    | $S$                          |
| zero padding              | $P$                          |
| **Output**                |                              |
| output size               | $W_2\times H_2\times K$      |
| output width              | $W_2=\frac{W_1-F+2P}{S}+1\\$ |
| output height             | $H_2=\frac{H_1-F+2P}{S}+1\\$ |
| **Parameters**            |                              |
| parameter number (weight) | $F^2CK$                      |
| parameter number (bias)   | $K$                          |

### Pooling layer

Make the representations smaller and more manageable.

**An example of max pooling:**

<img src="pics\5-pooling.png" style="zoom:60%;"/>

| **Input**      |                           |
| -------------- | ------------------------- |
| image size     | $W_1\times H_1\times C$   |
| spatial extent | $F\times F$               |
| stride         | $S$                       |
| **Output**     |                           |
| output size    | $W_2\times H_2\times C$   |
| output width   | $W_2=\frac{W_1-F}{S}+1\\$ |
| output height  | $H_2=\frac{H_1-F}{S}+1\\$ |

### Convolutional Neural Networks (CNN)

CNN stack CONV, POOL, FC layers.

**CNN Trends:**

1. Smaller filters and deeper architectures.
2. Getting rid of POOL/FC layers (just CONV).

**Historically architectures of CNN looked like:**

<img src="pics\5-model_history.png" style="zoom:40%;"/>

where usually $m$ is large, $0\le n\le5$,  $0\le k\le2$.

Recent advances such as **ResNet** / **GoogLeNet** have challenged this paradigm.