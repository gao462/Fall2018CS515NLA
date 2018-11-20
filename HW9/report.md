# Homework 9

- Jianfei Gao
- 0029986102
- **Problem 0**: I did not discuss with anyone except those discussions on Piazza. All text and code which are necessary are included.

## Problem 1

$$
\begin{aligned}
\begin{bmatrix}
A & \underline{y} \\
\underline{z}^\text{T} & \theta \\
\end{bmatrix}
\begin{bmatrix}
\underline{x}' \\
\gamma \\
\end{bmatrix} &=
\begin{bmatrix}
\underline{b} \\
\beta \\
\end{bmatrix} \\
\begin{bmatrix}
A & \underline{y} \\
\underline{z}^\text{T} & \theta \\
\end{bmatrix}
\begin{bmatrix}
\underline{x}' \\
\gamma - \frac{\beta}{\theta} \\
\end{bmatrix} &=
\begin{bmatrix}
\underline{b} \\
0 \\
\end{bmatrix} \\
\begin{bmatrix}
A & \underline{y} \\
\frac{1}{\theta}\underline{z}^\text{T} & 1 \\
\end{bmatrix}
\begin{bmatrix}
\underline{x}' \\
\gamma - \frac{\beta}{\theta} \\
\end{bmatrix} &=
\begin{bmatrix}
\underline{b} \\
0 \\
\end{bmatrix} \\
\end{aligned}
$$

For the last form, it is the matrix version of Sherman Morrison, thus
$$
\begin{aligned}
\begin{bmatrix}
A & \underline{y} \\
\frac{1}{\theta}\underline{z}^\text{T} & 1 \\
\end{bmatrix}
\begin{bmatrix}
\underline{x}' \\
\gamma - \frac{\beta}{\theta} \\
\end{bmatrix} &=
\begin{bmatrix}
\underline{b} \\
0 \\
\end{bmatrix} \\
\gamma - \frac{\beta}{\theta} &=
-(1 - \frac{1}{\theta}\underline{z}^\text{T}A^{-1}\underline{y})^{-1}\frac{1}{\theta}\underline{z}^\text{T}A^{-1}\underline{b} \\
x' &=
A^{-1}\left[b - \left(\gamma - \frac{\beta}{\theta}\right)\underline{y}\right] \\
\end{aligned}
$$

To summary, we can solve the new enlarged system by following steps.
$$
\begin{aligned}
\underline{p} &= A^{-1}\underline{y} \quad (\text{Solve a system with $A$}) \\
\underline{q} &= A^{-1}\underline{b} \quad (\text{Solve a system with $A$}) \\
\nu &= (1 - \frac{1}{\theta}\underline{z}^\text{T}\underline{p})^{-1}\frac{1}{\theta}\underline{z}^\text{T}\underline{q} \\
\gamma &= \frac{\beta}{\theta} - \nu \\
\underline{x}' &= A^{-1}(b + \nu\underline{y})  \quad (\text{Solve a system with $A$}) \\
\end{aligned}
$$

## Problem 2

Define $L\underline{v} = \underline{u}$.
$$
\begin{aligned}
A + \underline{u}\underline{u}^\text{T} &= LDL^\text{T} + L\underline{v}\underline{v}^\text{T}L^\text{T} \\
&= L(D + \underline{v}\underline{v}^\text{T})L^\text{T}
\end{aligned}
$$
Since there is no permission for $D + \underline{v}\underline{v}^\text{T}$ being diagonal, we assume $D + \underline{v}\underline{v}^\text{T} = L_1D_1L_1^\text{T}$. Thus
$$
L'D'L'^\text{T} = (LL_1)D_1(LL_1)^\text{T}
$$
Based on [Citation 2.1](http://stanford.edu/group/SOL/papers/ggms74.pdf), there is a $O(n^2)$ method to solve.
```python
def cholesky_rank1_update(L0, D0, u):
    n = L0.shape[0]
    D1 = D0.copy()
    L1 = L0.copy()
    a = 1
    w = u
    for j in range(n):
        p = w[j, 0]
        D1[j, j] = D0[j, j] + a * p * p
        b = p * a / D1[j, j]
        a = D0[j, j] * a / D1[j, j]
        for r in range(j + 1, n):
            w[r, 0] = w[r, 0] - p * L0[r, j]
            L1[r, j] = L0[r, j] + b * w[r, 0]
    return L1, D1
```

A test code is attached in Appendix 1.

## Problem 3

1. Given an arbitary $\underline{c}​$, we assume that
   $$
   x = \frac{\sum\limits_{i = 0}^{k}{c_{i}A^{i}b}}{\|\sum\limits_{i = 0}^{k}{c_{i}A^{i}b}\|}
   $$

   Thus, we want to maximize
   $$
   \begin{aligned}
   \lambda(\underline{c}) = \frac{x^\text{T}Ax}{x^\text{T}x}
   &= \frac{\left(\sum\limits_{i = 0}^{k}{c_{i}A^{i}b}\right)^\text{T}A\left(\sum\limits_{i = 0}^{k}{c_{i}A^{i}b}\right)}{\left(\sum\limits_{i = 0}^{k}{c_{i}A^{i}b}\right)^\text{T}\left(\sum\limits_{i = 0}^{k}{c_{i}A^{i}b}\right)} \\
   &= \frac{\sum\limits_{i = 0}^{k}{\sum\limits_{j = 0}^{k}{c_{i}c_{j}b^\text{T}(A^{i})^\text{T}A^{j + 1}b}}}{\sum\limits_{i = 0}^{k}{\sum\limits_{j = 0}^{k}{c_{i}c_{j}b^\text{T}(A^{i})^\text{T}A^{j}b}}} \\
   \end{aligned}
   $$
   Define
   $$
   \begin{aligned}
   \alpha_{i, j} &= b^\text{T}(A^{i})^\text{T}A^{j+1}b \\
   \beta_{i, j} &= b^\text{T}(A^{i})^\text{T}A^{j}b \\
   f(\underline{c}) &= \sum\limits_{i = 0}^{k}{\sum\limits_{j = 0}^{k}{c_{i}c_{j}\alpha_{i, j}}} \\
   g(\underline{c}) &= \sum\limits_{i = 0}^{k}{\sum\limits_{j = 0}^{k}{c_{i}c_{j}\beta_{i, j}}} \\
   \end{aligned}
   $$
   We can use **gradient ascend algorithm** to maximize $\lambda(\underline{c})$ over $\underline{c}$, and we just pick all-$\frac{1}{k + 1}$ vector as the initial guess.

   We will have following gradients
   $$
   \begin{aligned}
   \lambda(\underline{c}) &= \frac{f(\underline{c})}{g(\underline{c})} \\
   \frac{\partial \lambda(\underline{c})}{\partial c_{i}}
   &= \frac{\frac{\partial f(\underline{c})}{\partial c_{i}} g(\underline{c}) - f(\underline{c}) \frac{\partial g(\underline{c})}{\partial c_{i}}}{g(\underline{c})^2} \\
   \frac{\partial f(\underline{c})}{\partial c_{i}} &=
   \sum\limits_{j = 0}^{k}{c_{j}(\alpha_{i, j} + \alpha_{j, i})} \\
   \frac{\partial g(\underline{c})}{\partial c_{i}} &=
   \sum\limits_{j = 0}^{k}{c_{j}(\beta_{i, j} + \beta_{j, i})} \\
   \end{aligned}
   $$

   Each time, we can update $\underline{c}$ by
   $$
   c_{i} = c_{i} + \frac{\partial \lambda(\underline{c})}{\partial c_{i}}
   $$
   We update $\underline{c}$ 10000 times, and take the largest result from them.

   The test code is given in Appendix 2.

2. Here is a log-error over steps plot on several symmetric positive definite matrices.

   ![p3](.\p3.png)

3. For some cases, it may not improve the results.

## Appendix 1

```python
import numpy as np
from scipy.linalg import ldl
np.random.seed(1234567890)

def cholesky_rank1_update(L0, D0, u):
    n = L0.shape[0]
    D1 = D0.copy()
    L1 = L0.copy()
    a = 1
    w = u
    for j in range(n):
        p = w[j, 0]
        D1[j, j] = D0[j, j] + a * p * p
        b = p * a / D1[j, j]
        a = D0[j, j] * a / D1[j, j]
        for r in range(j + 1, n):
            w[r, 0] = w[r, 0] - p * L0[r, j]
            L1[r, j] = L0[r, j] + b * w[r, 0]
    return L1, D1

N = 5
A = np.random.normal(size=(N, N))
A = A @ A.T
u = np.random.normal(size=(N, 1))
L, D, _ = ldl(A)
L1, D1, _ = ldl(A + u @ u.T)
L2, D2 = cholesky_rank1_update(L, D, u)
print(np.linalg.norm(L1 - L2))
print(np.linalg.norm(D1 - D2))
```

## Appendix 2

```python
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
np.random.seed(1234567890)

def krylov_power_method(A, max_iter=1000):
    k = A.shape[0] - 1
    alpha = np.zeros(shape=(k + 1, k + 1))
    beta = np.zeros(shape=(k + 1, k + 1))
    b = np.zeros(shape=(k + 1, 1))
    b[k, 0] = 1
    basis = [b]
    for i in range(k):
        basis.append(A @ basis[-1])
    for i in range(k + 1):
        for j in range(k + 1):
            alpha[i, j] = basis[i].T @ A @ basis[j] 
            beta[i, j] = basis[i].T @ basis[j]
    c = [1 / (k + 1) for i in range(k + 1)]
    dc = [0 for i in range(k + 1)]
    cnt = 0
    best_rho = None
    rhos = []
    improve = True
    for cnt in range(max_iter):
        f = 0
        g = 0
        for i in range(k + 1):
            for j in range(k + 1):
                f += (c[i] * c[j] * alpha[i, j])
                g += (c[i] * c[j] * beta[i, j])
        rho = f / g
        rhos.append(rho)
        if best_rho is None or rho > best_rho:
            best_rho = rho
        else:
            pass
        for i in range(k + 1):
            df = 0
            dg = 0
            for j in range(k + 1):
                df += (c[j] * (alpha[i, j] + alpha[j, i]))
                dg += (c[j] * (beta[i, j] + beta[j, i]))
            dc[i] = (df * g - f * dg) / (g * g)
        for i in range(k + 1):
            c[i] += dc[i]
        cnt += 1
    return best_rho, rhos

error_curve = {
    4: None,
    5: None,
    7: None,
    10: None,
}
for n in error_curve:
    A = np.random.normal(size=(n, n))
    A = A @ A.T
    real_rho = np.linalg.eig(A)[0].max()
    rho, rhos = krylov_power_method(A, max_iter=10000)
    print("{:2d} {:15.8f} {:15.8f}".format(n, real_rho, rho))
    error_curve[n] = np.fabs(np.array(rhos) - real_rho)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
indices = np.arange(1, 10000 + 1)
for n in error_curve:
    ax.plot(indices, np.log(error_curve[n]), label="{}".format(n))
ax.legend(title=r"k + 1")
ax.set_xlabel('#Iterations')
ax.set_ylabel('Log-Error')
fig.savefig('p3.png')
```

