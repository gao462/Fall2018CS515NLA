# Homework 1

- Jianfei Gao
- 0029986102

In the following work, I will use Python 3 with `numpy`, `scipy` and `matplotlib` for code works.
It should have `image1.png`, `image2.png`, and `image3.png` under the same folder of the code to run.

## Problem 1

- Sub Question 1
  
  $$
  \begin{bmatrix}
    1 & 1 & 2 \\
    3 & 5 & 8 \\
    13 & 21 & 34 \\
  \end{bmatrix}
  \begin{bmatrix}
    1 & -2 & 3 \\
    -4 & 5 & -6 \\
    7 & -8 & 9 \\
  \end{bmatrix}
  = \begin{bmatrix}
    11 & -13 & 15 \\
    39 & -45 & 51 \\
    167 & -193 & 219 \\
  \end{bmatrix}
  $$

- Sub Question 2
  
  $$
  x^\text{T}y = 500500
  $$
  
  He is Gauss.

- Sub Question 3

  $$
  \begin{aligned}
  e x^\text{T} &= \begin{bmatrix}
      1.5 & 2 & -3 \\
      1.5 & 2 & -3 \\
      1.5 & 2 & -3 \\
      1.5 & 2 & -3 \\
  \end{bmatrix} \\
  x e^\text{T} &= \begin{bmatrix}
      1.5 & 1.5 & 1.5 & 1.5 \\
      2 & 2 & 2 & 2 \\
      -3 & -3 & -3 & -3 \\
  \end{bmatrix}
  \end{aligned}
  $$

- Sub Question 4

  $$
  \begin{aligned}
  e_1 x^\text{T} &= \begin{bmatrix}
      -5 & 4 & 2 \\
      0 & 0 & 0 \\
      0 & 0 & 0 \\
  \end{bmatrix} \\
  x e_3^\text{T} &= \begin{bmatrix}
      0 & 0 & -5 \\
      0 & 0 & 4 \\
      0 & 0 & 2 \\
  \end{bmatrix}
  \end{aligned}
  $$

## Problem 2

- Sub Question 1

  $$
  \begin{aligned}
  \begin{bmatrix}
      1 & a \\
      0 & 1 \\
  \end{bmatrix}
  \begin{bmatrix}
    1 & -a \\
    0 & 1 \\
  \end{bmatrix}
  = \begin{bmatrix}
    1 + 0 & -a + a \\
    0 + 0 & 0 + 1 \\
  \end{bmatrix}
  = \begin{bmatrix}
    1 & 0 \\
    0 & 1 \\
  \end{bmatrix}
  \end{aligned}
  $$

- Sub Question 2

  $$
  \begin{aligned}
  \begin{bmatrix}
      \textbf{I} & \textbf{A} \\
      0 & \textbf{I} \\
  \end{bmatrix}
  \begin{bmatrix}
      \textbf{I} & -\textbf{A} \\
      0 & \textbf{I} \\
  \end{bmatrix}
  = \begin{bmatrix}
    \textbf{I}^2 + 0 & -\textbf{I}\textbf{A} + \textbf{A}\textbf{I} \\
    0 + 0 & 0 + \textbf{I}^2 \\
  \end{bmatrix}
  = \begin{bmatrix}
    \textbf{I} & 0 \\
    0 & \textbf{I} \\
  \end{bmatrix}
  \end{aligned}
  $$

- Sub Question 3

  $$
  \begin{aligned}
  \begin{bmatrix}
      \textbf{A} & \textbf{B} \\
      0 & \textbf{C} \\
  \end{bmatrix}
  \begin{bmatrix}
      \textbf{A}^{-1} & -\textbf{A}^{-1}\textbf{B}\textbf{C}^{-1} \\
      0 & \textbf{C}^{-1} \\
  \end{bmatrix}
  = \begin{bmatrix}
    \textbf{A}\textbf{A}^{-1} + 0 & -\textbf{A}\textbf{A}^{-1}\textbf{B}\textbf{C}^{-1} + \textbf{B}\textbf{C}^{-1} \\
    0 + 0 & 0 + \textbf{C}\textbf{C}^{-1} \\
  \end{bmatrix}
  = \begin{bmatrix}
    \textbf{I} & 0 \\
    0 & \textbf{I} \\
  \end{bmatrix}
  \end{aligned}
  $$

  Then, the inverse will be

  $$
  \begin{bmatrix}
      \textbf{A} & \textbf{B} \\
      0 & \textbf{C} \\
  \end{bmatrix}^{-1}
  = \begin{bmatrix}
      \textbf{A}^{-1} & -\textbf{A}^{-1}\textbf{B}\textbf{C}^{-1} \\
      0 & \textbf{C}^{-1} \\
  \end{bmatrix}
  $$

## Problem 3

- Sub Question 1

  We know that

  $$
  \begin{aligned}
  f_{a, i} &= \begin{cases}
    f_i & a = i \\
    0   & a \neq i \\
  \end{cases} \\
  g_{i, b} &= \begin{cases}
    g_i & b = i \\
    0   & b \neq i \\
  \end{cases}
  \end{aligned} \\
  $$

  Thus

  $$
  \begin{aligned}
  \textbf{F}\textbf{G}
  &= \begin{bmatrix}
    f_{1,1} & f{1,2} & \cdots & f{1,n} \\
    f_{2,1} & f{2,2} & \cdots & f{2,n} \\
    \vdots  & \vdots & \ddots & \vdots \\ 
    f_{n,1} & f{n,2} & \cdots & f{n,n} \\
  \end{bmatrix}
  \begin{bmatrix}
    g_{1,1} & g{1,2} & \cdots & g{1,n} \\
    g_{2,1} & g{2,2} & \cdots & g{2,n} \\
    \vdots  & \vdots & \ddots & \vdots \\ 
    g_{n,1} & g{n,2} & \cdots & g{n,n} \\
  \end{bmatrix} \\
  &= \begin{bmatrix}
    \sum\limits_{i = 1}^{n}{f_{1, i}g_{i,1}} & \sum\limits_{i = 1}^{n}{f_{1, i}g_{i,2}} & \cdots & \sum\limits_{i = 1}^{n}{f_{1, i}g_{i,n}} \\
    \sum\limits_{i = 1}^{n}{f_{2, i}g_{i,1}} & \sum\limits_{i = 1}^{n}{f_{2, i}g_{i,2}} & \cdots & \sum\limits_{i = 1}^{n}{f_{2, i}g_{i,n}} \\
    \vdots  & \vdots & \ddots & \vdots \\
    \sum\limits_{i = 1}^{n}{f_{n, i}g_{i,1}} & \sum\limits_{i = 1}^{n}{f_{n, i}g_{i,2}} & \cdots & \sum\limits_{i = 1}^{n}{f_{n, i}g_{i,n}} \\
  \end{bmatrix} \\
  &= \begin{bmatrix}
    \sum\limits_{i = 1}^{n}{f_{i}g_{i}} & 0 & \cdots & 0 \\
    0 & \sum\limits_{i = 1}^{n}{f_{i}g_{i}} & \cdots & 0 \\
    \vdots  & \vdots & \ddots & \vdots \\ 
    0 & 0 & \cdots & \sum\limits_{i = 1}^{n}{f_{i}g_{i}} \\
  \end{bmatrix} \\
  
  \textbf{G}\textbf{F}
  &= 
  \begin{bmatrix}
    g_{1,1} & g{1,2} & \cdots & g{1,n} \\
    g_{2,1} & g{2,2} & \cdots & g{2,n} \\
    \vdots  & \vdots & \ddots & \vdots \\ 
    g_{n,1} & g{n,2} & \cdots & g{n,n} \\
  \end{bmatrix}
  \begin{bmatrix}
    f_{1,1} & f{1,2} & \cdots & f{1,n} \\
    f_{2,1} & f{2,2} & \cdots & f{2,n} \\
    \vdots  & \vdots & \ddots & \vdots \\ 
    f_{n,1} & f{n,2} & \cdots & f{n,n} \\
  \end{bmatrix} \\
  &= \begin{bmatrix}
    \sum\limits_{i = 1}^{n}{g_{1, i}f_{i,1}} & \sum\limits_{i = 1}^{n}{g_{1, i}f_{i,2}} & \cdots & \sum\limits_{i = 1}^{n}{g_{1, i}f_{i,n}} \\
    \sum\limits_{i = 1}^{n}{g_{2, i}f_{i,1}} & \sum\limits_{i = 1}^{n}{g_{2, i}f_{i,2}} & \cdots & \sum\limits_{i = 1}^{n}{g_{2, i}f_{i,n}} \\
    \vdots  & \vdots & \ddots & \vdots \\
    \sum\limits_{i = 1}^{n}{g_{n, i}f_{i,1}} & \sum\limits_{i = 1}^{n}{g_{n, i}f_{i,2}} & \cdots & \sum\limits_{i = 1}^{n}{g_{n, i}f_{i,n}} \\
  \end{bmatrix} \\
  &= \begin{bmatrix}
    \sum\limits_{i = 1}^{n}{f_{i}g_{i}} & 0 & \cdots & 0 \\
    0 & \sum\limits_{i = 1}^{n}{f_{i}g_{i}} & \cdots & 0 \\
    \vdots  & \vdots & \ddots & \vdots \\ 
    0 & 0 & \cdots & \sum\limits_{i = 1}^{n}{f_{i}g_{i}} \\
  \end{bmatrix} \\
  \end{aligned}
  $$

  So
  
  $$
  \textbf{F}\textbf{G} = \textbf{G}\textbf{F}
  $$

- Sub Question 2

  $$
  \begin{aligned}
  q(x) &= \text{Diag}((\textbf{I} - \textbf{H})x) \cdot (\textbf{I} + \textbf{H})x \\
  &= \text{Diag}(x - \textbf{H}x) \cdot (x + \textbf{H}x) \\       
  &= (\text{Diag}(x) - \text{Diag}(\textbf{H}x)) \cdot (x + \textbf{H}x) \\
  &= \text{Diag}(x)x - \text{Diag}(\textbf{H}x)x + \text{Diag}(x)\textbf{H}x - \text{Diag}(\textbf{H}x)\textbf{H}x \\
  \end{aligned}
  $$

  We can show that

  $$
  \begin{aligned}
  \text{Diag}(\textbf{H}x)x
  &= \begin{bmatrix}
    \sum\limits_{i = 1}^{n}{h_{1,i}x_i} & 0 & \cdots & 0 \\
    0 & \sum\limits_{i = 1}^{n}{h_{2,i}x_i} & \cdots & 0 \\
    \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & \cdots & \sum\limits_{i = 1}^{n}{h_{n,i}x_i} \\
  \end{bmatrix}
  \begin{bmatrix}
    x_1 \\
    x_2 \\
    \vdots \\
    x_n \\
  \end{bmatrix} \\
  &= \begin{bmatrix}
    \sum\limits_{i = 1}^{n}{h_{1,i}x_ix_1} \\
    \sum\limits_{i = 1}^{n}{h_{2,i}x_ix_2} \\
    \vdots \\
    \sum\limits_{i = 1}^{n}{h_{n,i}x_ix_n} \\
  \end{bmatrix} \\
  
  \text{Diag}(x)\textbf{H}x
  &= \begin{bmatrix}
    x_1 & 0 & \cdots & 0 \\
    0 & x_2 & \cdots & 0 \\
    \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & \cdots & x_n \\
  \end{bmatrix}
  \begin{bmatrix}
    \sum\limits_{i = 1}^{n}{h_{1,i}x_i} \\
    \sum\limits_{i = 1}^{n}{h_{2,i}x_i} \\
    \vdots \\
    \sum\limits_{i = 1}^{n}{h_{n,i}x_i} \\
  \end{bmatrix} \\
  &= \begin{bmatrix}
    \sum\limits_{i = 1}^{n}{h_{1,i}x_ix_1} \\
    \sum\limits_{i = 1}^{n}{h_{2,i}x_ix_2} \\
    \vdots \\
    \sum\limits_{i = 1}^{n}{h_{n,i}x_ix_n} \\
  \end{bmatrix}
  \end{aligned}
  $$

  So
  
  $$
  \begin{aligned}
  q(x) &= \text{Diag}(x)x  - \text{Diag}(\textbf{H}x)\textbf{H}x - (\text{Diag}(\textbf{H}x)x - \text{Diag}(x)\textbf{H}x) \\
  &= \text{Diag}(x)x - \text{Diag}(\textbf{H}x)\textbf{H}x
  \end{aligned}
  $$

- Sub Question 3

  $$
  \begin{aligned}
  y^\text{T}q(x) &= y^\text{T} (\text{Diag}(x)x - \text{Diag}(\textbf{H}x)\textbf{H}x) \\
  &= (yx^{-1}x)^\text{T} (\text{Diag}(x) - \text{Diag}(\textbf{H}x)\textbf{H})x \\
  &= x^\text{T} \left[ (yx^{-1})^\text{T}(\text{Diag}(x) - \text{Diag}(\textbf{H}x)\textbf{H}) \right] x
  \end{aligned} \\
  $$

  So

  $$
  \textbf{C} = (yx^{-1})^\text{T}(\text{Diag}(x) - \text{Diag}(\textbf{H}x)\textbf{H})
  $$

## Problem 4

- Sub Question 1

  $$
  A = \begin{bmatrix}
     1 & -1 &  0 &  0 &
    -1 &  1 &  0 &  0 &
     0 &  0 &  0 &  0 &
     0 &  0 &  0 &  0 \\
     0 &  0 &  1 & -1 &
     0 &  0 & -1 &  1 &
     0 &  0 &  0 &  0 &
     0 &  0 &  0 &  0 \\
     0 &  0 &  0 &  0 &
     0 &  0 &  0 &  0 &
     1 & -1 &  0 &  0 &
    -1 &  1 &  0 &  0 \\
     0 &  0 &  0 &  0 &
     0 &  0 &  0 &  0 &
     0 &  0 &  1 & -1 &
     0 &  0 & -1 &  1 \\
  \end{bmatrix}
  $$

- Sub Question 2

  ```bash
  52.443138644099236
  ```

- Sub Question 3

  Reshaping is just taking every four elements as a new column for a matrix in the Julia example in this question. Thus, it can generate a vector index matrix with the same shape as image telling the vector index of each pixel.

- Sub Question 4

  All the non-zero entries in the first row will be 1 or -1, and they are

  $$
  \begin{aligned}
  W2[1,1] &= 1 \\
  W2[1,2] &= -1 \\
  W2[1,17] &= -1 \\
  W2[1,18] &= 1 \\
  \end{aligned}
  $$

- Sub Question 5

  ```bash
  0.08235294
  1.04313727
  0.73333336
  ```

  Based on the given original images, these results can be regarded as scores of having a line in the image. The higher score the result gives, the higher probability of having a line the image will have.

- Sub Question 6

  ```bash
  0.082353
  1.28235289
  0.86666657
  ```

- Sub Question 7

  See appendix 2.

- Sub Question 8

  - Sparse Matrix (`scipy.sparse.lil_matrix`): 0.000228 seconds

  - Dense Matrix (`numpy.matrixlib.defmatrix.matrix`): 0.000021 seconds

  - Different sparse matrix classes have different efficiencies, but I think `lil_matrix` is equivalent to that of Julia.

## Appendix 1 (Code of Problem 1)

```python
import numpy as np


# sub question 1
def sub1():
    print()
    a = np.array([
        [1, 1, 2],
        [3, 5, 8],
        [13, 21, 34],
    ])
    b = np.array([
        [1, -2, 3],
        [-4, 5, -6],
        [7, -8, 9],
    ])
    print(a @ b)

# sub question 2
def sub2():
    print()
    x = np.ones((1000, 1))
    y = np.arange(1, 1000 + 1)[: , np.newaxis]
    print(x.T @ y)

# sub question 3
def sub3():
    print()
    x = np.array([1.5, 2, -3])[: , np.newaxis]
    e = np.ones((4, 1))
    print(x.shape, e.shape)
    print(e @ x.T)
    print(x @ e.T)

# sub question 4
def sub4():
    print()
    x = np.array([-5, 4, 2])[: , np.newaxis]
    e_1 = np.zeros((3, 1))
    e_1[0] = 1
    e_3 = np.zeros((3, 1))
    e_3[2] = 1
    print(e_1 @ x.T)
    print(x @ e_3.T)

# run sub questions
sub1()
sub2()
sub3()
sub4()
```

## Appendix 2 (Code of Problem 4)


```python
import time
from matplotlib.image import imread
import numpy as np
import scipy.sparse as sparse


# load data
X1 = imread('image1.png').astype('float64')
X2 = imread('image2.png').astype('float64')
X3 = imread('image3.png').astype('float64')


# build a weight matrix
def crude_edge_detector(nin, nout, base):
    vecs = []
    for i in range(nin // 2):
        for j in range(nin // 2):
            mx = np.zeros((nin, nin))
            mx[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] = base
            vecs.append(mx.reshape(-1))
    return np.array(vecs)

# build a sparse weight matrix
def sparse_crude_edge_detector(nin, nout, base):
    mx = sparse.lil_matrix((nout ** 2, nin ** 2))
    cnt = 0
    indices = np.arange(nin ** 2).reshape(nin, nin)
    for i in range(nin // 2):
        for j in range(nin // 2):
            i0, j0 = i * 2, j * 2
            for iv, jv in ((0, 0), (0, 1), (1, 0), (1, 1)):
                it = i0 + iv
                jt = j0 + jv
                vid = indices[it][jt]
                val = base[iv][jv]
                mx[cnt, vid] = val
            cnt += 1
    return mx

# build a dense weight matrix
def dense_crude_edge_detector(nin, nout, base):
    return sparse_crude_edge_detector(nin, nout, base).todense()


# buid a neural network
class Network(object):
    def __init__(self, base, type=None):
        if type is None:
            func = crude_edge_detector
        elif type == 'sparse':
            func = sparse_crude_edge_detector
        elif type == 'dense':
            func = dense_crude_edge_detector
        else:
            func = None
        self.W1 = crude_edge_detector(32, 16, base)
        self.W2 = crude_edge_detector(16, 8, base)
        self.W3 = np.ones((1, 8 * 8))

    def relu(self, x):
        x[x < 0] = 0
        return x

    def forward(self, x):
        x = self.relu(self.W1 @ x)
        x = self.relu(self.W2 @ x)
        x = self.relu(self.W3 @ x)
        return x

    def __call__(self, x):
        return self.forward(x)


# sub question 2
def sub2():
    print()
    print(np.trace(X1 + X2 + X3))

# sub question 4
def sub4():
    print()
    base = np.array([[1, -1], [-1, 1]])
    W1 = crude_edge_detector(32, 16, base)
    W2 = crude_edge_detector(16, 8, base)
    indices = np.where(W2[0] != 0)[0]
    vals = W2[0, indices]
    for idx, val in zip(indices, vals):
        print(1, idx + 1, val)

# sub question 5
def sub5():
    print()
    base = np.array([[1, -1], [-1, 1]])
    net = Network(base)
    print(net(X1.T.reshape(-1)))
    print(net(X2.T.reshape(-1)))
    print(net(X3.T.reshape(-1)))

# sub question 6
def sub6():
    print()
    base = np.array([[-1, 1], [1, -1]])
    net = Network(base)
    print(net(X1.T.reshape(-1)))
    print(net(X2.T.reshape(-1)))
    print(net(X3.T.reshape(-1)))

# sub question 7
def sub7():
    print()
    base = np.array([[1, -1], [-1, 1]])
    W1 = crude_edge_detector(32, 16, base)
    sW1 = sparse_crude_edge_detector(32, 16, base)
    dW1 = dense_crude_edge_detector(32, 16, base)
    print((sW1 == W1).all())
    print((dW1 == W1).all())
    print((sW1 == dW1).all())

# sub question 8
def sub8():
    print()
    X = X1.T.reshape(-1)
    base = np.array([[1, -1], [-1, 1]])
    sW1 = sparse_crude_edge_detector(32, 16, base)
    dW1 = dense_crude_edge_detector(32, 16, base)
    stime = 99
    dtime = 99
    for i in range(100):
        timer = time.time()
        _ = sW1 @ X
        stime = min(stime, time.time() - timer)
        timer = time.time()
        _ = dW1 @ X
        dtime = min(dtime, time.time() - timer)
    print("Sparse: {:.6f} sec".format(stime))
    print("Dense : {:.6f} sec".format(dtime))

# run sub questions
sub2()
sub4()
sub5()
sub6()
sub7()
sub8()
```