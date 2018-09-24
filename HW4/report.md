# Homework 4

- Jianfei Gao
- 0029986102
- **Problem 0**: I did not discuss with anyone except those discussions on Piazza. All text and code which are necessary are included.
- I also use `pytorch` library to save tables.

## Problem 1

To minimize, we need
$$
\begin{aligned}
\left[ \frac{\partial f(x_{k + 1})}{\partial c_i} \right]_i &= 0 \\
\left[ \frac{\partial f(x_{k + 1})}{\partial c_j} \right]_j &= 0 \\
\end{aligned}
$$

For the first formula, we should have
$$
\begin{aligned}
\left[ A x_{k + 1} - b \right]_i &= 0 \\
\left[ A (x_k + c_i e_i + c_j e_j) - b \right]_i &= 0 \\
A_{i,i} c_i &= - \left[ Ax_k - b \right]_i - c_j \left[ A e_j \right]_i \\
A_{i,i} c_i &= - \left[ g_k \right]_i - A_{i,j} c_j \\
c_i &= -\frac{1}{A_{i,i}} (\left[ g_k \right]_i + A_{i,j} c_j).
\end{aligned}
$$

Then, symmetrically, for the second formula, we should have
$$
\begin{aligned}
c_j &= -\frac{1}{A_{j,j}} (\left[ g_k \right]_j + A_{j,i} c_i).
\end{aligned}
$$

The update rule will be
$$
\begin{aligned}
x_{k + 1} &= x_{k} + c_i e_i + c_j e_j \\
&= x_{k} - \frac{\left[ g_k \right]_i + A_{i,j} c_j}{A_{i,i}} e_i - \frac{\left[ g_k \right]_j + A_{j,i} c_i}{A_{j,j}} e_j.
\end{aligned}
$$

## Problem 2

The reformation are following:
$$
\begin{aligned}
m_k &= A g_k & \text{(The Only Matrix-Vector Multiplication)}\\
\alpha_k &= \frac{g_k^\text{T}g_k}{g_k^\text{T}m_k} \\
x_{k + 1} &= x_{k} - \alpha_{k} g_k \\
g_{k + 1} &= A (x_{k} - \alpha_k g_k) - b \\
&= (Ax_{k} - b) - \alpha_k (Ag_k) \\
&= g_{k} - \alpha_{k} m_{k} \\
\end{aligned}
$$


See Appendix 1.

## Problem 3

See Appendix 1.

## Problem 4

For $n = 40$, the table is too large (100MB) to upload, so I only include plots here. You can still get the table by running the code.
According to my estimation, it takes about $10$ seconds for $n = 10$, then it should take $10 \times \left( \frac{20}{10}^2 \right)^2 = 160$ seconds (3 minutes) for $n = 20$ and $10 \times \left( \frac{40}{10}^2 \right)^2 = 2560$ seconds (45 minutes) for $n = 40$. This is my Numpy implementation, and I have no priori how long should Julia take.

![p4](C:\Users\gao46\Documents\Linux\Workplace\CS515NLA\HW4\p4.png)

## Problem 5

The relative residual increases for all the three methods at the first several works. And the cyclic coordinate descent is no longer faster than gradient descent.

![p5](C:\Users\gao46\Documents\Linux\Workplace\CS515NLA\HW4\p5.png)

## Appendix 1

```python
import numpy as np
import scipy.sparse as sparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
np.random.seed(1234567890)


# +--------------------------------------------------
# | Laplacian From HW2
# +--------------------------------------------------


def laplacian(N, f):
    idx_mx = np.arange(1, (N + 1) ** 2 + 1).reshape(N + 1, N + 1)
    ops = ((-1, 0, 1), (0, -1, 1), (0, 0, -4), (1, 0, 1), (0, 1, 1))
    fvec = np.zeros(((N + 1) ** 2, 1))
    liv = []
    for i in range(N + 1):
        for j in range(N + 1):
            fvec[idx_mx[i, j] - 1, 0] = f(i / N, j / N)
            if i == 0 or i == N or j == 0 or j == N:
                liv.append([idx_mx[i, j], idx_mx[i, j], -4])
            else:
                for (dx_i, dx_j, val) in ops:
                    liv.append([idx_mx[i, j], idx_mx[i + dx_i, j + dx_j], val])
    A = np.zeros(((N + 1) ** 2, (N + 1) ** 2), dtype=int)
    for i, j, v in liv:
        A[i - 1, j - 1] = v
    return sparse.csr_matrix(A), fvec


# +--------------------------------------------------
# | CandyLand From HW2
# +--------------------------------------------------


def candy_land():
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
    candyland_matrix = read_csv('candyland-matrix.csv')

    def matrix_from_liv(shape, liv):
        mx = np.zeros(shape)
        for i, j, val in liv:
            mx[int(i) - 1][int(j) - 1] = val
        return mx
    trans_mx = matrix_from_liv((140, 140), candyland_matrix)

    i_mx = np.eye(140)
    b = np.ones((140, 1))
    b[133, 0] = 0
    return i_mx - trans_mx.T, b


# +--------------------------------------------------
# | Descent Solver
# +--------------------------------------------------


# graident descent
def gradient_descent(A, b, thres=1e-4):
    x = b.copy()
    g = A @ x - b
    num_works = 0
    record = []
    while True:
        loss = np.linalg.norm(g) / np.linalg.norm(b)
        record.append((num_works, loss))
        if loss < thres:
            return x, np.array(record)
        else:
            pass
        m = A @ g
        alpha = (g.T @ g) / (g.T @ m)
        x = x - alpha * g
        g = g - alpha * m
        num_works += (A != 0).sum()

# coordinate descent
def cyclic_coordinate_descent(A, b, thres=1e-4):
    x = b.copy()
    num_works = 0
    record = []
    while True:
        for i in range(len(b)):
            g = A @ x - b
            loss = np.linalg.norm(g) / np.linalg.norm(b)
            record.append((num_works, loss))
            if loss < thres:
                return x, np.array(record)
            else:
                pass
            g_i = A[i, :] @ x - b[i, 0]
            x[i, 0] = x[i, 0] - g_i / A[i, i]
            num_works += (A[i, :] != 0).sum()

def random_coordinate_descent(A, b, thres=1e-4):
    x = b.copy()
    num_works = 0
    record = []
    while True:
        i = np.random.randint(0, len(b))
        g = A @ x - b
        loss = np.linalg.norm(g) / np.linalg.norm(b)
        record.append((num_works, loss))
        if loss < thres:
            return x, np.array(record)
        else:
            pass
        g_i = A[i, :] @ x - b[i, 0]
        x[i, 0] = x[i, 0] - g_i / A[i, i]
        num_works += (A[i, :] != 0).sum()


# draw plot and save table
def plot_and_table(records, title, save):
    fig, axes = plt.subplots(1, len(records), figsize=(8 * len(records), 6))
    fig.suptitle(title)
    if len(records) == 1:
        axes = [axes]
    else:
        pass
    for i, n in enumerate(records):
        axes[i].set_title(r"Matrix Size {}".format(n))
        axes[i].set_xlabel(r'Number Of Working $A$ Non-zeros')
        axes[i].set_ylabel(r'$Log_{10}$ Relative Residual')
        axes[i].set_ylim(-5, 1)
        for method in records[n]:
            num_works = records[n][method][:, 0]
            loss = records[n][method][:, 1]
            log_loss = np.log10(loss)
            axes[i].plot(num_works, log_loss, label=method)
        axes[i].legend()
    fig.savefig(save)


# problem 4
def p4():
    print()
    records = {}
    for n in (10, 20, 40):
        A, b = laplacian(n, lambda x, y: 1)
        print(A.shape)
        records[len(b)] = {}
        print('Gradient Descent')
        x, record = gradient_descent(A, b, thres=1e-4)
        records[len(b)]['Gradient Descent'] = record
        x, record = cyclic_coordinate_descent(A, b, thres=1e-4)
        print('Cyclic Coordinate Descent')
        records[len(b)]['Cyclic Coordinate Descent'] = record
        print('Random Coordinate Descent')
        x, record = random_coordinate_descent(A, b, thres=1e-4)
        records[len(b)]['Random Coordinate Descent'] = record
    plot_and_table(records, 'Laplacian', 'p4.png')
    torch.save(records, 'p4.pt')

# problem 5
def p5():
    print()
    records = {}
    A, b = candy_land()
    print(A.shape)
    records[len(b)] = {}
    print('Gradient Descent')
    x, record = gradient_descent(A, b, thres=1e-4)
    records[len(b)]['Gradient Descent'] = record
    x, record = cyclic_coordinate_descent(A, b, thres=1e-4)
    print('Cyclic Coordinate Descent')
    records[len(b)]['Cyclic Coordinate Descent'] = record
    print('Random Coordinate Descent')
    x, record = random_coordinate_descent(A, b, thres=1e-4)
    records[len(b)]['Random Coordinate Descent'] = record
    plot_and_table(records, 'Candy Land', 'p5.png')
    torch.save(records, 'p5.pt')

# run problems
p4()
p5()
```