# Homework 10

- Jianfei Gao
- 0029986102
- **Problem 0**: I did not discuss with anyone except those discussions on Piazza. All text and code which are necessary are included.

## Problem 1

1. See Appendix 1

2. Based on the Julia code, I think that we are looking for **relative residuals rather than residuals**. Since the results are in different level, I use table rather than plot to show them. Since I implement CG by myself in Python, so I also do a simple comparison between my implementation (My Impl) and a translation from Julia to Python (JL2PY).

   | Iteration | MINRES | CG (My Impl) | CG2 (JL2PY) |
   | - | - | - | - |
   | 1  | 0.98994949 | 7.00000000  | 7.00000000  |
   | 2  | 0.97979590 | 6.85857128  | 6.85857128  |
   | 3  | 0.96953597 | 6.71714225  | 6.71714225  |
   | 4  | 0.95916630 | 6.57571289  | 6.57571289  |
   | 5  | 0.94868330 | 6.43428318  | 6.43428318  |
   | 6  | 0.93808315 | 6.29285309  | 6.29285309  |
   | 7  | 0.92736185 | 6.15142260  | 6.15142260  |
   | 8  | 0.91651514 | 6.00999168  | 6.00999168  |
   | 9  | 0.90553851 | 5.86856030  | 5.86856030  |
   | 10 | 0.89442719 | 5.72712843  | 5.72712843  |
   | 11 | 0.88317609 | 5.58569602  | 5.58569602  |
   | 12 | 0.87177979 | 5.44426304  | 5.44426304  |
   | 13 | 0.86023253 | 5.30282943  | 5.30282943  |
   | 14 | 0.84852814 | 5.16139516  | 5.16139516  |
   | 15 | 0.83666003 | 5.01996016  | 5.01996016  |
   | 16 | 0.82462113 | 4.87852437  | 4.87852437  |
   | 17 | 0.81240384 | 4.73708771  | 4.73708771  |
   | 18 | 0.80000000 | 4.59565012  | 4.59565012  |
   | 19 | 0.78740079 | 4.45421149  | 4.45421149  |
   | 20 | 0.77459667 | 4.31277173  | 4.31277173  |
   | 21 | 0.76157731 | 4.17133072  | 4.17133072  |
   | 22 | 0.74833148 | 4.02988834  | 4.02988834  |
   | 23 | 0.73484692 | 3.88844442  | 3.88844442  |
   | 24 | 0.72111026 | 3.74699880  | 3.74699880  |
   | 25 | 0.70710678 | 3.60555128  | 3.60555128  |

3. It takes **50** iterations. My guess is that **number of iterations to converge $x$ will have the same tendency as $n$**. I also do some numerical experiments, and from those results, I also guess that
   $$
   x = \lceil \frac{n}{2} \rceil
   $$



## Problem 2

1. The values should always be -20 which means that $V^\text{T}V = I$, but both cases actually diverge from that. For the two vectors, random and $\underline{e}/\sqrt{n}$, the results are independent from initial vector. All the following code will be located in Appendix 2.

   ![](C:\Users\gao46\Documents\Linux\Workplace\CS515NLA\HW10\p2-1.png)

2. Here are the plots.

   ![](C:\Users\gao46\Documents\Linux\Workplace\CS515NLA\HW10\p2-2.png)

3. For random vector, $\beta_{31} = 9.006461531139257$. For $\underline{e}/\sqrt{n}$, $\beta_{31} = 19.119662713392128$. It should be 0.

4. Here is the plot.

   ![](C:\Users\gao46\Documents\Linux\Workplace\CS515NLA\HW10\p2-4.png)

## Problem 3

Based on the construction of A, we will have
$$
\begin{aligned}
\begin{bmatrix}
2 & 1 & \cdots & 1 \\
1 & 2 & \cdots & 1 \\
\vdots & \vdots & \ddots & \vdots \\
1 & 1 & \cdots & 2 \\
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n \\
\end{bmatrix}
&= \begin{bmatrix}
x_1 + \sum\limits_{i=1}^{n}{x_{i}} \\
x_2 + \sum\limits_{i=1}^{n}{x_{i}} \\
\vdots \\
x_n + \sum\limits_{i=1}^{n}{x_{i}} \\
\end{bmatrix}
= \begin{bmatrix}
b_1 \\
b_2 \\
\vdots \\
b_n \\
\end{bmatrix} \\
(n + 1) \sum\limits_{i=1}^{n}{x_{i}} &= \sum\limits_{i=1}^{n}{b_{i}} \\
x_{i} &= b_{i} - \frac{1}{n + 1} \sum\limits_{j=1}^{n}{b_{j}} \\
s &= \sum\limits_{j=1}^{n}{b_{j}} \quad \text{(Define)} \\
\underline{x} &= \underline{b} - \frac{s}{n + 1}\underline{e} \\
\end{aligned}
$$
So, $\underline{x}$ will be a linear combination of at most two vectors, so 2 iterations will be enough for MINRES to converge.

I put code in Appendix 3 to verify this.

## Problem 4

1. We show that it is equivalent to solve $A^\text{T}A\underline{x} = A^\text{T}\underline{b}$.
   $$
   \begin{aligned}
   \begin{bmatrix}
   I & A \\
   A^\text{T} & 0 \\
   \end{bmatrix}
   \begin{bmatrix}
   \underline{r} \\
   \underline{x} \\
   \end{bmatrix}
   &= \begin{bmatrix}
   \underline{b} \\
   0 \\
   \end{bmatrix} \\
   \underline{r} + A\underline{x} &= \underline{b} \\
   A^\text{T}\underline{r} &= 0 \\
   A^\text{T}\underline{r} + A^\text{T}A\underline{x} &= A^\text{T}\underline{b} \\
   A^\text{T}A\underline{x} &= A^\text{T}\underline{b} \quad \text{(Solve system)} \\
   \underline{r} &= \underline{b} - A\underline{x} \\
   \end{aligned}
   $$

2. Code is in the Appendix 4.

   I count the number of iterations, then compute matrix-vector multiplication manually. I ignore the matrix-vector multiplication for computing relative residuals.

   Here are the numbers of iterations:

   | Neumann | MINRES | CG   |
   | ------- | ------ | ---- |
   | 383     | 45     | 45   |

   For Neumann
   $$
   \begin{aligned}
   \underline{r}^{(0)} &= \underline{b} \\
   \underline{x}^{(0)} &= b \\
   \underline{r}^{(i + 1)} &= (I - A)\underline{r}^{(i)} \quad \text{(Matrix-Vector)} \\
   \underline{x}^{(i + 1)} &= \underline{x}^{(i)} + \underline{r}^{(i + 1)}
   \end{aligned}
   $$
   So, it takes $383 \times 1 = 383$ matrix-vector multiplications.

   For MINRES and CG
   $$
   \begin{aligned}
   \underline{v}^{(0)} &= 0 \\
   \beta_1 &= \|\underline{b}\| \\
   \underline{v}^{(1)} &= \frac{1}{\beta_1}\underline{b} \\
   \underline{w} &= A\underline{v}^{(i)} \quad \text{(Matrix-Vector)} \\
   \alpha_{i} &= {\underline{v}^{(i)}}^\text{T}\underline{w} \\
   \underline{w} &= \underline{w} - \alpha_{i}\underline{v}^{(i)} - \beta_{i}\underline{v}^{(i - 1)} \\
   \beta_{i + 1} &= \|\underline{w}\| \\
   \underline{v}^{(i + 1)} &= \frac{1}{\beta_{i + 1}}\underline{w} \\
   V_{i + 1} &= \begin{bmatrix} V_{i} & \underline{v}^{(i + 1)} \end{bmatrix} \\
   T_{i + 1} &= \begin{bmatrix}
   T_{i} & \beta_{i}\underline{e}_{i - 1} + \alpha_{i}\underline{e}_{i} \\
   0 & \beta_{i + 1} \\
   \end{bmatrix} \\
   T_{i + 1} \underline{y}^{(i)} &= \|\underline{b}\|\underline{e}_1 \quad \text{(Solve tridiagonal-plus system)} \\
   \underline{x}^{*} &= V_{i}\underline{y}^{*} \quad \text{(Matrix-Vector)} \\
   \end{aligned}
   $$
   For solving tridiagonal-plus system (tridiagonal matrix $\tilde{T}_{i + 1}$ with an additional tail row) by MINRES (Minimize least-square residual) or CG (Solve tridiagonal matrix $\tilde{T}_{i + 1}$), they all have methods to fill in each value which is not related to matrix-vector multiplication, but they are all $O\left((i + 1)^2\right)$ for $T_{i + 1}$ ([Citation](https://www.researchgate.net/publication/226708527_An_inversion_algorithm_for_general_tridiagonal_matrix)) which is same as matrix-vector multiplication complexity. So, I regard it as a matrix-vector multiplication.

   So, they both takes $45 \times 2 + 1 = 91$ matrix-vector multiplications.

   Here are the numbers of matrix-vector multiplications:

   | Neumann | MINRES | CG   |
   | ------- | ------ | ---- |
   | 383     | 91     | 91   |


## Appendix 1

```python
import re
import numpy as np
np.random.seed(1234567890)


# generate A
def spddiagm(n):
    mx = np.zeros(shape=(n, n))
    for i in range(n):
        mx[i, i] = 4
    for i in range(n - 1):
        mx[i, i + 1] = -2
        mx[i + 1, i] = -2
    return mx

# lanczos
def lanczos(A, b, k):
    # allocate matrices
    assert k > 1
    n = b.shape[0]
    V_mx = np.zeros(shape=(n, k))
    T_mx = np.zeros(shape=(k, k - 1))

    # allocate buffer
    v = [None for i in range(k)]
    alpha = [None for i in range(k - 1)]
    beta = [None for i in range(k)]

    # initial loop is special
    beta[0] = np.linalg.norm(b)
    v[0] = b / beta[0]
    alpha[0] = float(v[0].T @ A @ v[0])
    w = A @ v[0] - alpha[0] * v[0]
    beta[1] = np.linalg.norm(w)
    v[1] = w / beta[1]

    # loop for the remaining part
    for i in range(1, k - 1):
        alpha[i] = float(v[i].T @ A @ v[i])
        w = A @ v[i] - alpha[i] * v[i] - beta[i] * v[i - 1]
        beta[i + 1] = np.linalg.norm(w)
        v[i + 1] = w / beta[i + 1]

    # fill in values
    for i in range(k):
        V_mx[:, [i]] = v[i]
    for i in range(k - 1):
        if i > 0:
            T_mx[i - 1, i] = beta[i]
        else:
            pass
        T_mx[i, i] = alpha[i]
        T_mx[i + 1, i] = beta[i + 1]
    return V_mx, T_mx

# minimum residual method
def minres(A, b, iter=25):
    residuals = []
    for i in range(2, iter + 2):
        V, T = lanczos(A, b, k=i)
        e = np.zeros(shape=(T.shape[0], 1))
        e[0, 0] = 1
        b2 = np.linalg.norm(b) * e
        A2 = T
        y = np.linalg.lstsq(A2, b2, rcond=None)[0]
        rres = np.linalg.norm(b - A @ V[:, 0:-1] @ y) / np.linalg.norm(b)
        residuals.append(rres)
    return np.array(residuals)

# conjugate gradient method
def cg(A, b, iter=25):
    residuals = []
    for i in range(2, iter + 2):
        V, T = lanczos(A, b, k=i)
        e = np.zeros(shape=(T.shape[0] - 1, 1))
        e[0, 0] = 1
        b2 = np.linalg.norm(b) * e
        A2 = T[0:-1, :]
        y = np.linalg.solve(A2, b2)
        rres = np.linalg.norm(b - A @ V[:, 0:-1] @ y) / np.linalg.norm(b)
        residuals.append(rres)
    return np.array(residuals)

def cg2(A, b, iter=25, tol=1e-8):
    x = np.zeros(shape=b.shape)
    r = b.copy()
    rho_1 = 0
    residuals = []
    for i in range(iter):
        z = r
        rho = float(r.T @ z)
        if i > 0:
            beta = rho / rho_1
            p = z + beta * p
        else:
            p = z
        q = A @ p
        alpha = rho / float(p.T @ q)
        x = x + alpha * p
        r = r - alpha * q
        rres = np.linalg.norm(r) / np.linalg.norm(b)
        residuals.append(rres)
        if rres <= tol:
            break
        else:
            rho_1 = rho
    return np.array(residuals)

def cg_tol(A, b, tol=1e-8):
    x = np.zeros(shape=b.shape)
    r = b.copy()
    rho_1 = 0
    residuals = []
    cnt = 0
    while cnt < 10000:
        cnt += 1
        z = r
        rho = float(r.T @ z)
        if cnt > 1:
            beta = rho / rho_1
            p = z + beta * p
        else:
            p = z
        q = A @ p
        alpha = rho / float(p.T @ q)
        x = x + alpha * p
        r = r - alpha * q
        rres = np.linalg.norm(r) / np.linalg.norm(b)
        residuals.append(rres)
        if rres <= tol:
            break
        else:
            rho_1 = rho
    return cnt, np.array(residuals)

# sub question 1
def sub1():
    print()
    A = spddiagm(100)
    b = np.ones(shape=(100, 1))
    r1 = minres(A, b, iter=25)
    r2 = cg(A, b, iter=25)
    r3 = cg2(A, b, iter=25)
    for i in range(25):
        print("| {:<2d} | {:<11.8f} | {:<11.8f} | {:<11.8f} |".format(i + 1, r1[i], r2[i], r3[i]))

# sub question 2
def sub2():
    print()
    A = spddiagm(100)
    b = np.ones(shape=(100, 1))
    cnt, _ = cg_tol(A, b, tol=1e-8)
    print(cnt)

    buffer = []
    for n in (10, 20, 40, 45, 46, 47, 48, 80):
        A = spddiagm(n)
        b = np.ones(shape=(n, 1))
        cnt, _ = cg_tol(A, b, tol=1e-8)
        buffer.append(cnt)
    print(buffer)

# run sub questions
sub1()
sub2()
```

## Appendix 2

```python
import re
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234567890)


# generate A
def strako(n=30, lambda_1=0.1, lambda_n=100, rho=0.9):
    mx = np.zeros(shape=(n, n))
    for i in range(n):
        d = lambda_1 + i / (n - 1) * (lambda_n - lambda_1) * (rho ** (n - i - 1))
        mx[i, i] = d
    return mx

# lanczos
def lanczos(A, b, k):
    # allocate matrices
    assert k > 1
    n = A.shape[0]
    V_mx = np.zeros(shape=(n, k))
    T_mx = np.zeros(shape=(k, k - 1))

    # allocate buffer
    v = [None for i in range(k)]
    alpha = [None for i in range(k - 1)]
    beta = [None for i in range(k)]

    # initial loop is special
    beta[0] = np.linalg.norm(b)
    v[0] = b / beta[0]
    alpha[0] = float(v[0].T @ A @ v[0])
    w = A @ v[0] - alpha[0] * v[0]
    beta[1] = np.linalg.norm(w)
    v[1] = w / beta[1]

    # loop for the remaining part
    for i in range(1, k - 1):
        alpha[i] = float(v[i].T @ A @ v[i])
        w = A @ v[i] - alpha[i] * v[i] - beta[i] * v[i - 1]
        beta[i + 1] = np.linalg.norm(w)
        v[i + 1] = w / beta[i + 1]
    return v, alpha, beta

def fillin(v, alpha, beta, n, k=None):
    # allocate matrices
    assert k is None or k > 1
    k = k or len(v)
    V_mx = np.zeros(shape=(n, k))
    T_mx = np.zeros(shape=(k, k - 1))

    # fill in values
    for i in range(k):
        V_mx[:, [i]] = v[i]
    for i in range(k - 1):
        if i > 0:
            T_mx[i - 1, i] = beta[i]
        else:
            pass
        T_mx[i, i] = alpha[i]
        T_mx[i + 1, i] = beta[i + 1]
    return V_mx, T_mx

# sub question 1
def sub1():
    print()
    n = 30
    A = strako(n)
    init_vec0 = np.random.normal(size=(n, 1))
    init_vec1 = np.ones(shape=(n, 1)) / np.sqrt(n)
    lanczos_vecs0 = lanczos(A=A, b=init_vec0, k=31)
    lanczos_vecs1 = lanczos(A=A, b=init_vec1, k=31)

    V = lanczos_vecs0[0][0]
    quant0 = [np.linalg.norm(V.T @ V - np.eye(1))]
    V = lanczos_vecs1[0][0]
    quant1 = [np.linalg.norm(V.T @ V - np.eye(1))]
    quant2 = [0]
    for i in range(1, n):
        V, _ = fillin(*lanczos_vecs0, n=n, k=i + 1)
        quant0.append(np.linalg.norm(V.T @ V - np.eye(i + 1)))
        V, _ = fillin(*lanczos_vecs1, n=n, k=i + 1)
        quant1.append(np.linalg.norm(V.T @ V - np.eye(i + 1)))
        quant2.append(0)
    quant0 = np.log10(np.array(quant0) + 1e-20)
    quant1 = np.log10(np.array(quant1) + 1e-20)
    quant2 = np.log10(np.array(quant2) + 1e-20)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(np.arange(1, 31), quant0, color='red'  , alpha=0.85, linestyle='-' , label='Random')
    ax.plot(np.arange(1, 31), quant1, color='blue' , alpha=0.85, linestyle='-' , label=r'$\mathbf{e}/\sqrt{n}$')
    ax.plot(np.arange(1, 31), quant2, color='green', alpha=0.85, linestyle='--', label='SHOULD')
    ax.legend()
    fig.savefig('p2-1.png')

# sub question 2
def sub2():
    print()
    n = 30
    A = strako(n)
    init_vec0 = np.random.normal(size=(n, 1))
    init_vec1 = np.ones(shape=(n, 1)) / np.sqrt(n)
    lanczos_vecs0 = lanczos(A=A, b=init_vec0, k=31)
    lanczos_vecs1 = lanczos(A=A, b=init_vec1, k=31)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    quant0 = []
    quant1 = []
    for i in range(0, n):
        v = lanczos_vecs0[0]
        quant0.append(float(v[0].T @ v[i]))
        v = lanczos_vecs1[0]
        quant1.append(float(v[0].T @ v[i]))
    quant0 = np.log10(np.fabs(np.array(quant0)) + 1e-20)
    quant1 = np.log10(np.fabs(np.array(quant1)) + 1e-20)

    ax1.plot(np.arange(1, 31), quant0, color='red'  , alpha=0.85, linestyle='-' , label='Random')
    ax1.plot(np.arange(1, 31), quant1, color='blue' , alpha=0.85, linestyle='-' , label=r'$\mathbf{e}/\sqrt{n}$')
    ax1.legend()

    quant0 = []
    quant1 = []
    for i in range(2, n):
        v = lanczos_vecs0[0]
        quant0.append(float(v[i - 2].T @ v[i]))
        v = lanczos_vecs1[0]
        quant1.append(float(v[i - 2].T @ v[i]))
    quant0 = np.log10(np.fabs(np.array(quant0)) + 1e-20)
    quant1 = np.log10(np.fabs(np.array(quant1)) + 1e-20)

    ax2.plot(np.arange(3, 31), quant0, color='red'  , alpha=0.85, linestyle='-' , label='Random')
    ax2.plot(np.arange(3, 31), quant1, color='blue' , alpha=0.85, linestyle='-' , label=r'$\mathbf{e}/\sqrt{n}$')
    ax2.legend()

    fig.savefig('p2-2.png')

# sub question 3
def sub3():
    print()
    n = 30
    A = strako(n)
    init_vec0 = np.random.normal(size=(n, 1))
    init_vec1 = np.ones(shape=(n, 1)) / np.sqrt(n)
    lanczos_vecs0 = lanczos(A=A, b=init_vec0, k=31)
    lanczos_vecs1 = lanczos(A=A, b=init_vec1, k=31)
    print(lanczos_vecs0[2][n])
    print(lanczos_vecs1[2][n])

# sub question 3
def sub4():
    print()
    n = 30
    A = strako(n)
    init_vec0 = np.random.normal(size=(n, 1))
    init_vec1 = np.ones(shape=(n, 1)) / np.sqrt(n)
    lanczos_vecs0 = lanczos(A=A, b=init_vec0, k=61)
    lanczos_vecs1 = lanczos(A=A, b=init_vec1, k=61)

    quant0 = []
    quant1 = []
    for i in range(1, 61):
        V, T = fillin(*lanczos_vecs0, n=n, k=i + 1)
        quant0.append(np.linalg.norm(A @ V[:, 0:-1] - V @ T))
        V, T = fillin(*lanczos_vecs1, n=n, k=i + 1)
        quant1.append(np.linalg.norm(A @ V[:, 0:-1] - V @ T))
    quant0 = np.log10(np.array(quant0) + 1e-20)
    quant1 = np.log10(np.array(quant1) + 1e-20)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(np.arange(1, 61), quant0, color='red'  , alpha=0.85, linestyle='-' , label='Random')
    ax.plot(np.arange(1, 61), quant1, color='blue' , alpha=0.85, linestyle='-' , label=r'$\mathbf{e}/\sqrt{n}$')
    ax.legend()
    fig.savefig('p2-4.png')

# run sub questions
sub1()
sub2()
sub3()
sub4()
```

## Appendix 3

```python
import re
import numpy as np
np.random.seed(1234567890)


# generate A
def gen_A(n):
    e = np.ones(shape=(n, 1))
    return np.eye(n) + e @ e.T

# lanczos
def lanczos(A, b, k):
    # allocate matrices
    assert k > 1
    n = b.shape[0]
    V_mx = np.zeros(shape=(n, k))
    T_mx = np.zeros(shape=(k, k - 1))

    # allocate buffer
    v = [None for i in range(k)]
    alpha = [None for i in range(k - 1)]
    beta = [None for i in range(k)]

    # initial loop is special
    beta[0] = np.linalg.norm(b)
    v[0] = b / beta[0]
    alpha[0] = float(v[0].T @ A @ v[0])
    w = A @ v[0] - alpha[0] * v[0]
    beta[1] = np.linalg.norm(w)
    v[1] = w / beta[1]

    # loop for the remaining part
    for i in range(1, k - 1):
        alpha[i] = float(v[i].T @ A @ v[i])
        w = A @ v[i] - alpha[i] * v[i] - beta[i] * v[i - 1]
        beta[i + 1] = np.linalg.norm(w)
        v[i + 1] = w / beta[i + 1]

    # fill in values
    for i in range(k):
        V_mx[:, [i]] = v[i]
    for i in range(k - 1):
        if i > 0:
            T_mx[i - 1, i] = beta[i]
        else:
            pass
        T_mx[i, i] = alpha[i]
        T_mx[i + 1, i] = beta[i + 1]
    return V_mx, T_mx

# minimum residual method# minimum residual method
def minres(A, b, iter=25):
    residuals = []
    for i in range(2, iter + 2):
        V, T = lanczos(A, b, k=i)
        e = np.zeros(shape=(T.shape[0], 1))
        e[0, 0] = 1
        b2 = np.linalg.norm(b) * e
        A2 = T
        y = np.linalg.lstsq(A2, b2, rcond=None)[0]
        rres = np.linalg.norm(b - A @ V[:, 0:-1] @ y) / np.linalg.norm(b)
        residuals.append(rres)
    return np.array(residuals)

def minres_tol(A, b, tol=1e-8):
    residuals = []
    cnt = 0
    while cnt < 10000:
        cnt += 1
        V, T = lanczos(A, b, k=cnt + 1)
        e = np.zeros(shape=(T.shape[0], 1))
        e[0, 0] = 1
        b2 = np.linalg.norm(b) * e
        A2 = T
        y = np.linalg.lstsq(A2, b2, rcond=None)[0]
        rres = np.linalg.norm(b - A @ V[:, 0:-1] @ y) / np.linalg.norm(b)
        residuals.append(rres)
        if rres <= tol:
            break
        else:
            pass
    return cnt, np.array(residuals)

n = 200
A = gen_A(n)
b = np.random.normal(size=(n, 1))
x = b - b.sum() / (n + 1)
cnt, _ = minres_tol(A, b)
print(cnt)
```

## Appendix 4

```python
import re
import numpy as np
np.random.seed(1234567890)


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

# set A, b, and solve x = inv(A) @ b
A = np.eye(140) - trans_mx.T
b = np.ones((140, 1))
b[133, 0] = 0
A2 = A.T @ A
b2 = A.T @ b

# neumann
def neumann_tol(A, b, tol=1e-8):
    M = np.eye(A.shape[0]) - A
    r = b.copy()
    x = b.copy()
    cnt = 0
    while cnt < 10000:
        cnt += 1
        r = M @ r
        x = x + r
        if np.linalg.norm(b - A @ x) / np.linalg.norm(b) < tol:
            break
        else:
            pass
    return x, cnt

# lanczos
def lanczos(A, b, k):
    # allocate matrices
    assert k > 1
    n = b.shape[0]
    V_mx = np.zeros(shape=(n, k))
    T_mx = np.zeros(shape=(k, k - 1))

    # allocate buffer
    v = [None for i in range(k)]
    alpha = [None for i in range(k - 1)]
    beta = [None for i in range(k)]

    # initial loop is special
    beta[0] = np.linalg.norm(b)
    v[0] = b / beta[0]
    alpha[0] = float(v[0].T @ A @ v[0])
    w = A @ v[0] - alpha[0] * v[0]
    beta[1] = np.linalg.norm(w)
    v[1] = w / beta[1]

    # loop for the remaining part
    for i in range(1, k - 1):
        alpha[i] = float(v[i].T @ A @ v[i])
        w = A @ v[i] - alpha[i] * v[i] - beta[i] * v[i - 1]
        beta[i + 1] = np.linalg.norm(w)
        v[i + 1] = w / beta[i + 1]

    # fill in values
    for i in range(k):
        V_mx[:, [i]] = v[i]
    for i in range(k - 1):
        if i > 0:
            T_mx[i - 1, i] = beta[i]
        else:
            pass
        T_mx[i, i] = alpha[i]
        T_mx[i + 1, i] = beta[i + 1]
    return V_mx, T_mx

# minimum residual method# minimum residual method
def minres_tol(A, b, tol=1e-8):
    residuals = []
    cnt = 0
    while cnt < 10000:
        cnt += 1
        V, T = lanczos(A, b, k=cnt + 1)
        e = np.zeros(shape=(T.shape[0], 1))
        e[0, 0] = 1
        b2 = np.linalg.norm(b) * e
        A2 = T
        y = np.linalg.lstsq(A2, b2, rcond=None)[0]
        rres = np.linalg.norm(b - A @ V[:, 0:-1] @ y) / np.linalg.norm(b)
        residuals.append(rres)
        if rres <= tol:
            break
        else:
            pass
    return cnt, residuals

# conjugate gradient method
def cg_tol(A, b, tol=1e-8):
    residuals = []
    cnt = 0
    while cnt < 10000:
        cnt += 1
        V, T = lanczos(A, b, k=cnt + 1)
        e = np.zeros(shape=(T.shape[0] - 1, 1))
        e[0, 0] = 1
        b2 = np.linalg.norm(b) * e
        A2 = T[0:-1, :]
        y = np.linalg.solve(A2, b2)
        rres = np.linalg.norm(b - A @ V[:, 0:-1] @ y) / np.linalg.norm(b)
        residuals.append(rres)
        if rres <= tol:
            break
        else:
            pass
    return cnt, residuals

_, cnt = neumann_tol(A, b, tol=1e-8)
print(cnt)
cnt, residuals = minres_tol(A2, b2, tol=1e-8)
print(cnt)
cnt, residuals = cg_tol(A2, b2, tol=1e-8)
print(cnt)
```