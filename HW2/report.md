# Homework 2

- Jianfei Gao
- 0029986102
- **Problem 0**: I did not discuss with anyone except those discussions on Piazza. All text and code which are necessary are included.

## Problem 1

- Sub Question 1

  From the transition matrix given by `candyland-matrix.csv`, those sticky cells associated with 137, 138 and 139 are 47, 86 and 117.
  To detect a sticky cell $x$ from a transition matrix $T_{i, j}$ supposing that we only have transition rules, sticky cells and bridges introduced on the class, it should have following properties:
  1. It should only have probability 1 to transit to another cell $y$.
     $$
     T_{i, x} = \begin{cases}
     1 & i = y, y \neq x \\
     0 & \text{o.w.} \\
     \end{cases}
     $$

  2. Cell $y$ should be a special hidden cell specified for a sticky cell.

- Sub Question 2
  $$
  (I - T^\text{T})x = o^{(134)}
  $$
  where
  $$
  [o^{(x)}]_i = \begin{cases}
  0 & i = x \\
  1 & \text{o.w.} \\
  \end{cases}
  $$
  Then, we can get $x_{140} \approx 33.66804941$.

  Yes, there are some cells which have longer game. They are 6, 7, 8, 9, 10, 47. If we take hidden cells into consideration, they also include 137.

- Sub Question 3

  I stop adding terms when $||p_k|| < 10^{-10}$.
  $$
  \sum\limits_{k}{kT^{k - 1}t_{140}} \approx 33.66804941
  $$

- Sub Question 4
  $$
  \begin{aligned}
  \sum\limits_{k = 1}^{\infty}{kT^{k - 1}t_{140}}
  &= (I + 2T + 3T^2 + \cdots) t_{140} \\
  &= \left( \sum\limits_{k = 0}^{\infty}{T^k} + T \sum\limits_{k = 0}^{\infty}{T^k} + \cdots \right) t_{140} \\
  &= \left( \sum\limits_{k = 0}^{\infty}{T^k} \sum\limits_{k = 0}^{\infty}{T^k} \right) t_{140} \\
  &= (I - T)^{-2} t_{140} = (I - T)^{-2} T e_{140} \\
  
  x &= (I - T^\text{T})^{-1} o^{(134)} \\
  &= \sum\limits_{i = 0}^{\infty}{(T^\text{T})^i} o^{(134)} \\
  &= \left( \sum\limits_{i = 0}^{\infty}{T^i} \right)^\text{T} o^{(134)} \\
  &= \left( (I - T)^{-1} \right)^\text{T} o^{(134)} = \left( (I - T)^{-1} \right)^\text{T} (e - e_{134}) \\
  
  \left[ \sum\limits_{k = 1}^{\infty}{kT^{k - 1}t_{140}} \right]_{134}
  &= \left[ (I - T)^{-2} T e_{140} \right]_{134} \\
  &= \left[ (I - T)^{-2} T \right]_{134,140} \\
  
  x_{140} &= \left[ \left( (I - T)^{-1} \right)^\text{T} (e - e_{134}) \right]_{140} \\
  &= \left[ \left( (I - T)^{-1} \right)^\text{T} e \right]_{140} - \left[ \left( (I - T)^{-1} \right)^\text{T} e_{134} \right]_{140} \\
  &= \left[ \left( (I - T)^{-1} \right)^\text{T} e \right]_{140} - \left[ \left( (I - T)^{-1} \right)^\text{T} \right]_{140,134} \\
  &= \left[ \left( (I - T)^{-1} \right)^\text{T} e \right]_{140} - \left[ (I - T)^{-1} \right]_{134,140} \\
  
  \end{aligned}
  $$

  **Above are all the content related to Neumann series of a matrix in my opinion, but I still fails to get the proof**.

- Sub Question 5

  For sticky cells, there is no change. When you locate sticky cells 47, 86, 117, you will use a step to hidden cell 137, 138, 139 with probability 1 which simulates the first stick on corresponding cell. Then in the following steps, you will loop on hidden cells until you draw the same color card to exit.
  For bridge cells, the thing is that we count one more step for transitions "from 5 to 135", "from 35 to 136", "from 135 to 59" and "from 136 to 46". We just to exclude those extra steps from equation of sub question 2.
  $$
  (I - T^\text{T})x = o^{(\{134, 5, 35, 135, 136\})}
  $$
  where
  $$
  [o^{(X)}]_i = \begin{cases}
  0 & i \in X \\
  1 & \text{o.w.} \\
  \end{cases}
  $$
  Then, we can get $x_{140} \approx 33.01471025$.

  Yes, there are even more cells which have longer game. They are 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 47. If we take hidden cells into consideration, they also include 137.

## Problem 2

- Sub Question 1

  $$
  3^2 \begin{bmatrix}
  0 & 0 & 0 & 0 & 0 &  0 &  0 & 0 & 0 &  0 &  0 & 0 & 0 & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 &  0 &  0 & 0 & 0 &  0 &  0 & 0 & 0 & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 &  0 &  0 & 0 & 0 &  0 &  0 & 0 & 0 & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 &  0 &  0 & 0 & 0 &  0 &  0 & 0 & 0 & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 &  0 &  0 & 0 & 0 &  0 &  0 & 0 & 0 & 0 & 0 & 0 \\
  0 & 1 & 0 & 0 & 1 & -4 &  1 & 0 & 0 &  1 &  0 & 0 & 0 & 0 & 0 & 0 \\
  0 & 0 & 1 & 0 & 0 &  1 & -4 & 1 & 0 &  0 &  1 & 0 & 0 & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 &  0 &  0 & 0 & 0 &  0 &  0 & 0 & 0 & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 &  0 &  0 & 0 & 0 &  0 &  0 & 0 & 0 & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 &  1 &  0 & 0 & 1 & -4 &  1 & 0 & 0 & 1 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 &  0 &  1 & 0 & 0 &  1 & -4 & 1 & 0 & 0 & 1 & 0 \\
  0 & 0 & 0 & 0 & 0 &  0 &  0 & 0 & 0 &  0 &  0 & 0 & 0 & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 &  0 &  0 & 0 & 0 &  0 &  0 & 0 & 0 & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 &  0 &  0 & 0 & 0 &  0 &  0 & 0 & 0 & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 &  0 &  0 & 0 & 0 &  0 &  0 & 0 & 0 & 0 & 0 & 0 \\
  0 & 0 & 0 & 0 & 0 &  0 &  0 & 0 & 0 &  0 &  0 & 0 & 0 & 0 & 0 & 0 \\
  \end{bmatrix}
  \times \begin{bmatrix}
  u_{1} \\
  u_{2} \\
  u_{3} \\
  u_{4} \\
  u_{5} \\
  u_{6} \\
  u_{7} \\
  u_{8} \\
  u_{9} \\
  u_{10} \\
  u_{11} \\
  u_{12} \\
  u_{13} \\
  u_{14} \\
  u_{15} \\
  u_{16} \\
  \end{bmatrix}
  = \begin{bmatrix}
  0 \\
  0 \\
  0 \\
  0 \\
  0 \\
  9 (u_{2} + u_{5} - 4 u_{6} + u_{7} + u_{10}) \\
  9 (u_{3} + u_{6} - 4 u_{7} + u_{8} + u_{11}) \\
  0 \\
  0 \\
  9 (u_{6} + u_{9} - 4 u_{10} + u_{11} + u_{14}) \\
  9 (u_{7} + u_{10} - 4 u_{11} + u_{12} + u_{15}) \\
  0 \\
  0 \\
  0 \\
  0 \\
  0 \\
  0 \\
  \end{bmatrix}
  $$

- Sub Question 2

  See Appendix 2

- Sub Question 3

  ![p2](C:\Users\gao46\Documents\Linux\Workplace\CS515NLA\HW2\p2.png)

- Sub Question 4

  The infinite summation will diverge rather than converge. Neumann series will not working for this equation.

## Problem 3

See Appendix 3

## Appendix 1

```python
import numpy as np
import scipy


# load csv files
def read_csv(path):
    file = open(path)
    content = file.readlines()
    file.close()
    data = []
    for line in content:
        items = line.strip().split(',')
        items = [float(itr) for itr in items]
        data.append(items)
    return np.array(data)
candyland_cells = read_csv('candyland-cells.csv')
candyland_coords = read_csv('candyland-coords.csv')
candyland_matrix = read_csv('candyland-matrix.csv')

# create transition matrix from list of indices and values
# $T_{i, j}$: Probability from $j$ to $i$
def matrix_from_liv(shape, liv):
    mx = np.zeros(shape)
    for i, j, val in liv:
        mx[int(i) - 1][int(j) - 1] = val
    return mx
trans_mx = matrix_from_liv((140, 140), candyland_matrix)


# sub question 1
def sub1():
    print()
    for focus in (137, 138, 139):
        cond1 = (candyland_matrix[:, 0] == focus)
        cond2 = (candyland_matrix[:, 0] != candyland_matrix[:, 1])
        cond3 = (candyland_matrix[:, 2] == 1)
        for dst, src, prob in candyland_matrix[cond1 & cond2 & cond3]:
            print("{:3d} ---> {:3d} : {:.6f}".format(int(src), int(dst), prob))

# sub question 2
def sub2():
    print()
    i_mx = np.eye(140)
    b = np.ones((140, 1))
    b[133, 0] = 0
    x = np.linalg.inv(i_mx - trans_mx.T) @ b
    print(x[139])
    longers = list(np.where(x > x[139])[0])
    if len(longers) > 0:
        for i in longers:
            print("[{:3d}] = {}".format(i + 1, x[i, 0]))
    else:
        print('No Longer Starting')
    return x[139] # expect length of starting from i = 140

# sub question 3
def sub3():
    print()
    T = trans_mx
    b = np.zeros((140, 1))
    b[139, 0] = 1
    k = 1
    p = T @ b
    S = k * p
    EPSILON = 1e-10
    while True:
        k += 1
        p = T @ p
        if k * np.linalg.norm(p) < EPSILON:
            break
        else:
            S = S + k * p
    print(S[133])
    return S[133] # expect length of reaching i = 134 (starting from 140)

# sub question 4
def sub4(ex2, ex3):
    print()
    EPSILON = 1e-6
    if np.fabs(ex2 - ex3) < EPSILON:
        print("Sub 2 == Sub 3 ({})".format(EPSILON))
    else:
        print("Sub 2 != Sub 3 ({})".format(EPSILON))

# sub question 5
def sub5():
    print()
    i_mx = np.eye(140)
    b = np.ones((140, 1))
    b[133, 0] = 0
    b[134, 0] = 0
    b[135, 0] = 0
    b[4, 0] = 0
    b[34, 0] = 0
    x = np.linalg.inv(i_mx - trans_mx.T) @ b
    print(x[139])
    longers = list(np.where(x > x[139])[0])
    if len(longers) > 0:
        for i in longers:
            print("[{:3d}] = {}".format(i + 1, x[i, 0]))
    else:
        print('No Longer Starting')
    return x[139] # expect length of starting from i = 140

# run sub questions
sub1()
ex2 = sub2()
ex3 = sub3()
sub4(ex2, ex3)
sub5()
```

## Appendix 2

```python
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# sub question 1
def sub1():
    print()
    N = 3
    idx_mx = np.arange(1, (N + 1) ** 2 + 1).reshape(N + 1, N + 1)
    ops = ((-1, 0, 1), (0, -1, 1), (0, 0, -4), (1, 0, 1), (0, 1, 1))
    liv = []
    for i in range(N + 1):
        for j in range(N + 1):
            print("[{:2d}]: ({:1d}, {:1d})".format(idx_mx[i, j], i, j), end='')
            if i == 0 or i == N or j == 0 or j == N:
                print(' = 0')
            else:
                mx_op_str = []
                vec_op_str = []
                for (dx_i, dx_j, val) in ops:
                    mx_op_str.append("{:2d} * ({:1d}, {:1d})".format(
                                        val, i + dx_i, j + dx_j))
                    vec_op_str.append("{:2d} * [{:2d}]".format(
                                        val, idx_mx[i + dx_i, j + dx_j]))
                    liv.append([idx_mx[i, j], idx_mx[i + dx_i, j + dx_j], val])
                mx_op_str = ' + '.join(mx_op_str)
                vec_op_str = ' + '.join(vec_op_str)
                print(" = {} = {}".format(mx_op_str, vec_op_str))
    A = np.zeros(((N + 1) ** 2, (N + 1) ** 2), dtype=int)
    for i, j, v in liv:
        A[i - 1, j - 1] = v
    print(A)

# sub question 2
def laplacian(N, f):
    idx_mx = np.arange(1, (N + 1) ** 2 + 1).reshape(N + 1, N + 1)
    ops = ((-1, 0, 1), (0, -1, 1), (0, 0, -4), (1, 0, 1), (0, 1, 1))
    fvec = np.zeros(((N + 1) ** 2, 1))
    liv = []
    for i in range(N + 1):
        for j in range(N + 1):
            fvec[idx_mx[i, j] - 1, 0] = f(i / N, j / N)
            if i == 0 or i == N or j == 0 or j == N:
                pass
            else:
                for (dx_i, dx_j, val) in ops:
                    liv.append([idx_mx[i, j], idx_mx[i + dx_i, j + dx_j], val])
    A = np.zeros(((N + 1) ** 2, (N + 1) ** 2), dtype=int)
    for i, j, v in liv:
        A[i - 1, j - 1] = v
    return A, fvec

# sub question 3
def sub3():
    print()
    A, fvec = laplacian(10, lambda i, j: 1)
    uvec = np.linalg.lstsq(A, fvec, rcond=None)[0]
    N = int(np.sqrt(len(uvec))) - 1
    rows = np.tile(np.arange(0, N + 1), (N + 1, 1)).T
    cols = np.tile(np.arange(0, N + 1), (N + 1, 1))
    vals = uvec.reshape(N + 1, N + 1)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(rows, cols, vals, cmap=cm.YlGnBu)
    fig.savefig('p2.png')
    plt.show(fig)

# run sub questions
sub1()
sub3()
```

## Appendix 3

```python
import numpy as np
import scipy.sparse as sparse
np.random.seed(1234567890)


# get csc format from dense matrix
def dense_to_csc(mx):
    colptr = []
    rowptr = []
    nzval = []
    for cid in range(mx.shape[1]):
        colptr.append(len(rowptr))
        for rid in range(mx.shape[0]):
            if mx[rid, cid] != 0:
                rowptr.append(rid)
                nzval.append(mx[rid, cid])
            else:
                pass
    colptr.append(len(rowptr))
    return colptr, rowptr, nzval
mx = sparse.random(5, 6, density=0.4).todense()
colptr, rowval, nzval = dense_to_csc(mx)

# Sparse matrix-transpose multiplication by a vector
def csc_transpose_matvec(colptr, rowval, nzval, m, n, x):
    y = np.zeros((len(colptr) - 1, 1))
    for cid, (begin, end) in enumerate(zip(colptr[:-1], colptr[1:])):
        for rid, val in zip(rowval[begin:end], nzval[begin:end]):
            y[cid] += val * x[rid, 0]
    return y

# Row-inner-product
def csc_row_projection(colptr, rowval, nzval, m, n, i, x):
    y = np.zeros((1, 1))
    for cid, (begin, end) in enumerate(zip(colptr[:-1], colptr[1:])):
        for rid, val in zip(rowval[begin:end], nzval[begin:end]):
            if rid == i:
                y[0, 0] += val * x[cid, 0]
            else:
                pass
    return y

# Column-inner-product
def csc_column_projection(colptr, rowval, nzval, m, n, i, x):
    y = np.zeros((1, 1))
    begin, end = colptr[i], colptr[i + 1]
    for rid, val in zip(rowval[begin:end], nzval[begin:end]):
        y[0, 0] += val * x[rid, 0]
    return y

# sub question 1 test
def sub1():
    print()
    vec = np.random.normal(size=(5, 1))
    y1 = mx.T @ vec
    y2 = csc_transpose_matvec(colptr, rowval, nzval, None, None, vec)
    EPSILON = 1e-8
    print(np.fabs(y1 - y2).max() < EPSILON)

# sub question 2 test
def sub2():
    print()
    res = []
    for i in range(mx.shape[0]):
        vec = np.random.normal(size=(mx.shape[1], 1))
        y1 = mx[i, :] @ vec
        y2 = csc_row_projection(colptr, rowval, nzval, None, None, i, vec)
        EPSILON = 1e-8
        res.append(np.fabs(y1 - y2).max() < EPSILON)
    print(not False in res)

# sub question 3 test
def sub3():
    print()
    res = []
    for i in range(mx.shape[1]):
        vec = np.random.normal(size=(mx.shape[0], 1))
        y1 = mx[:, i].T @ vec
        y2 = csc_column_projection(colptr, rowval, nzval, None, None, i, vec)
        EPSILON = 1e-8
        res.append(np.fabs(y1 - y2).max() < EPSILON)
    print(not False in res)

# run sub question tests
sub1()
sub2()
sub3()
```