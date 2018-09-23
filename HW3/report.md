


# Homework 3

- Jianfei Gao
- 0029986102
- **Problem 0**: I do not discuss with anyone.

## Problem 1

- Sub Question 1

  For $n = 10​$ of $Au = f​$ of homework 2, we will find that all the eigenvalues of $A​$ are non-positive which means that $A​$ is a negative semi definite matrix. To get a positive semi definite matrix for Richardson method, we should solve $(-A)u = -f​$ instead, where all the eigenvalues of $(-A)​$ will be non-negative which means $(-A)​$ is  a positive semi definite matrix.

  For Richardson method, we then require $\alpha < \frac{2}{\rho(-A)}$ where $\rho(-A) \approx 7.804 < 8$. Thus, if we let $\alpha < \frac{1}{4}$, then it will have $\alpha < \frac{2}{8} < \frac{2}{\rho(-A)}$.

  So, we can let $\alpha = 0.25 \times 0.9 = 0.225$ to make Richardson method work.

- Sub Question 2

  $\alpha = 0.225$ still works because $\rho(-A) \approx 7.9508 < 8$.

- Sub Question 3

  We are going to solve $Au = f$ by

  $$
  u_{k + 1} = u_{k} + \alpha \left( (-f) - (-A) u_{k} \right).
  $$

  Then based on the code of Appendix 1, we can find that $\alpha = 0.25069657350751356$ will iterate 903 times to let relative residual less or equal than $10^{-5}$ for both $n = 10$ and $n = 20$.

- Sub Question 4

  For $n = 10$ and $n = 20$, it has already been shown in previous sub questions.

  Suppose $Au^{*} = f$, then

  $$
  \begin{aligned}
  u^{*} &= u^{*} + \alpha (A u^{*} - f) \\
  u_{k + 1} &= u_k + \alpha (A u_{k} - f) \\
  u^{*} - u_{k + 1} &= (u^{*} - u_{k}) + \alpha A (u^{*} - u_{k}) \\
  &= (I + \alpha A) (u^{*} - u_{k}) \\
  \end{aligned}
  $$

  Suppose $u^{*} - u_{k + 1} = e_{k + 1}$, $\alpha = \frac{1}{4} - \epsilon$, then we should have following equation

  $$
  \begin{bmatrix}
  [e_{k + 1}]_{1} \\
  \vdots \\
  [e_{k + 1}]_{n_1} \\
  [e_{k + 1}]_{n_2} \\
  \vdots \\
  [e_{k + 1}]_{n_3} \\
  [e_{k + 1}]_{n_4} \\
  \vdots \\
  [e_{k + 1}]_{n^2} \\
  \end{bmatrix}
  = \begin{bmatrix}
  4 \epsilon [e_{k}]_{1} \\
  \vdots \\
  4 \epsilon [e_{k}]_{n_1} \\
  4 \epsilon [e_{k}]_{n_2} + (\frac{1}{4} - \epsilon) [e_{k}]_{n_2 - 1} + (\frac{1}{4} - \epsilon) [e_{k}]_{n_2 + 1} + (\frac{1}{4} - \epsilon) [e_{k}]_{n_2 - n} + (\frac{1}{4} - \epsilon) [e_{k}]_{n_2 + n}\\
  \vdots \\
  4 \epsilon [e_{k}]_{n_3} + (\frac{1}{4} - \epsilon) [e_{k}]_{n_3 - 1} + (\frac{1}{4} - \epsilon) [e_{k}]_{n_3 + 1} + (\frac{1}{4} - \epsilon) [e_{k}]_{n_3 - n} + (\frac{1}{4} - \epsilon) [e_{k}]_{n_3 + n}\\
  4 \epsilon [e_{k}]_{n_4 + 1} \\
  \vdots \\
  4 \epsilon [e_{k}]_{n^2} \\
  \end{bmatrix}
  $$

  Based on this, and notice that each column with form $(\frac{1}{4} - \epsilon) [e_{k}]_{\square + b}$ ($b \in \{-n, -1, 1, n\}$) will not traverse all elements of $e_{k}$ ($\frac{4(n - 1)}{n^2}$ of them are zero out because of boundary conditions), we can observe first order norm of $e_{k + 1}$,

  $$
  \begin{aligned}
  |e_{k + 1}| &< 4 \epsilon |e_k| + 4 \times (\frac{1}{4} - \epsilon) |e_k| = |e_k|
  \end{aligned}
  $$

  Based on this observation, the iteration must converge.

## Problem 2

For any norm function $f(x)$ where $x \in \mathbb{R}^1$ which is indeed $f(x_1)$ where $x_1 \in \mathbb{R}$, it should satisfy following properties:

1. $f(x_1 + x_2) \leq f(x_1) + f(x_2)$
2. $f(ax_1) = |a|f(x_1)$
3. $f(x_1) = 0 \longleftrightarrow x_1 = 0$
4. $f(x_1) \geq 0$

Based on property 2, we will have $f(x_1 \times 1) = |x_1| f(1)$ for any real value $x_1$, which implies that $f(x_1) = \alpha |x_1|$ where $f(1) = \alpha$ is a constant.

## Problem 3

- Sub Question 1

  Obviously, $\max\limits_{i, j}{|A_{i, j}|} \geq 0$ because of the absolute operation. Let us check another three properties of any norm function $f(X)$ one by one.

  1. $f(A + B) \leq f(A) + f(B)$

     $$
     \begin{aligned}
     \max\limits_{i, j}{|(A + B)_{i, j}|} &= \max\limits_{i, j}{|A_{i, j} + B_{i, j}|} \\
     &\leq \max\limits_{i, j}{(|A|_{i, j} + |B|_{i, j})} \\
     &= \max\limits_{i, j}{|A|_{i, j}} + \max\limits_{i, j}{|B|_{i, j}} \\
     &= \max\limits_{i, j}{|A_{i, j}|} + \max\limits_{i, j}{|B_{i, j}|} \\
     \end{aligned}
     $$

  2. $f(\alpha A) = |\alpha| f(A)$

     $$
     \begin{aligned}
     \max\limits_{i, j}{|\alpha A_{i, j}|}
     = \max\limits_{i, j}{(|\alpha| \times |A_{i, j}|)}
     = |\alpha| \max\limits_{i, j}{|A_{i, j}|}
     \end{aligned}
     $$

  3. $f(A) = 0 \longleftrightarrow A = 0$

     If $\max\limits_{i, j}{|A_{i, j}|} = 0$, then $|A_{i, j}| = 0$ for any $(i, j)$, thus $A_{i, j} = 0$ for any $(i, j)$ which implies $A = 0$. The inverse side is quite obvious.

  Based on these 3 properties, $f(A) = \max\limits_{i, j}{|A_{i, j}|}$ is a norm function.

- Sub Question 2

  Let
  $$
  AB = \begin{bmatrix}
  1 & 2 \\
  3 & 4 \\
  \end{bmatrix}
  \begin{bmatrix}
  1 & 3 \\
  2 & 4 \\
  \end{bmatrix}
  = \begin{bmatrix}
  5 & 11 \\
  11 & 25 \\
  \end{bmatrix},
  $$
  then we will have
  $$
  \begin{aligned}
  f(A) &= 4 \\
  f(B) &= 4 \\
  f(AB) &= 25 > f(A) f(B) = 16 \\
  \end{aligned}
  $$
  which is a counter case.

- Sub Question 3

  Suppose $A$ is a $n \times m$ matrix, we then let $\sigma = nm > 0$, $g(A) = nmf(A) \geq 0$.

  Suppose another matrix $B$ is $m \times r$ matrix, then $g(B) = mrf(B)$, $g(AB) = nrf(AB)$.  Let us prove that this is sub-multiplicative by proving that $g(AB) \leq g(A)g(B)$.

  $$
  \begin{aligned}
  g(AB) &= \max\limits_{i, j}{nr|\sum\limits_{k = 1}^{m}{A_{ik}B_{kj}}|} \\
  &\leq nr \left( \max\limits_{i, j}{\sum\limits_{k = 1}^{m}{|A_{ik}B_{kj}|}} \right) \\
  &\leq nmr \left( \max\limits_{i, k}{|A_{ik}|} \times \max\limits_{k, j}{|B_{kj}|} \right) \\
  &\leq nm^2r \left( \max\limits_{i, k}{|A_{ik}|} \times \max\limits_{k, j}{|B_{kj}|} \right) \\
  &\leq nm \left( \max\limits_{i, k}{|A_{ik}|} \right) \times mr \left( \max\limits_{k, j}{|B_{kj}|} \right) \\
  &= g(A)g(B) \\
  \end{aligned}
  $$

  Then, prove again it is a matrix-norm.

  1. We have $f(A + B) \leq f(A) + f(B)$ and $\sigma > 0$, then $\sigma f(A + B) \leq \sigma (f(A) + f(B))$ which means $g(A + B) \leq g(A) + g(B)$.

  2. We have $f(\alpha A) = |\alpha| f(A)$ and $\sigma > 0$, then $\sigma f(\alpha A) = |\alpha| \sigma f(A)$ which means $g(\alpha A) = |\alpha| g(A)$.

  3. We have $f(A) = 0 \longleftrightarrow A = 0$ and $\sigma > 0$, then $g(A) = 0 \longleftrightarrow \sigma f(A) = 0 \longleftrightarrow f(A) = 0 \longleftrightarrow A = 0$.

  Based on above proves, $g(A) = nm f(A)$ is a sub-multiplicative matrix-norm.

## Problem 4

We can solve $(I - \alpha P) x = (1 - \alpha) v$ by

$$
x_{k + 1} = x_k + \eta \left( (1 - \alpha) v - (I - \alpha P) x_{k} \right)
$$

We show that $\eta = 1$ will be enough for this method to converge.

Suppose the solution $x^{*}$ satisfying $(I - \alpha P) x^{*} = (1 - \alpha) v$, then we will have
$$
\begin{aligned}
x^{*} &= x^{*} + \eta \left( (1 - \alpha) v - (I - \alpha P) x^{*} \right) \\
x_{k + 1} &= x_k + \eta \left( (1 - \alpha) v - (I - \alpha P) x_{k} \right) \\
x^{*} - x_{k + 1} &= (x^{*} - x_{k}) - \eta (I - \alpha P) (x^{*} - x_{k}) \\
&= (x^{*} - x_{k}) - (I - \alpha P) (x^{*} - x_{k}) \\
&= \alpha P (x^{*} - x_{k}) \\
\end{aligned}
$$

Let error of $k$-th iteration be $e_k = x^{*} - x_{k}$, then we will have
$$
e_{k + 1} = \alpha P e_{k}
$$

Let us observe the first order norm of $e_{k + 1}$,
$$
\begin{aligned}
|e_{k + 1}| &= \sum\limits_{i}{|[e_{k + 1}]_i|} = \alpha \sum\limits_{i}{|[Pe_{k}]_i|} \\
&= \alpha \sum\limits_{i}{\left| \sum\limits_{j}{P_{ij} [e_{k}]_j} \right|} \\
&\leq \alpha \sum\limits_{i}{\sum\limits_{j}{P_{ij} |[e_{k}]_j|}} \\
&= \alpha \sum\limits_{j}{\left( \sum\limits_{i} P_{ij} \right) |[e_{k}]_j|} \\
&= \alpha \sum\limits_{j}{|[e_{k}]_j|} = \alpha |e_{k}|
\end{aligned}
$$

Since $\alpha​$ should be a probability s.t. $0 < \alpha < 1​$, it is obvious that $|e_{k + 1}| \longrightarrow 0​$ which means $x_{k + 1} \longrightarrow x^{*}​$ when $k​$ is large enough. \\
So, Richardson method with $\eta = 1​$ will at least converge to the solution.

## Appendix 1

```python
import numpy as np

# laplacian construction from homework 2
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
    return A, fvec

# compute relative residual
def rres(A, b, x_head):
    return np.linalg.norm(A @ x_head - b) / np.linalg.norm(b)

# compute the largest absolute eigenvalue
def rho(A):
    return np.fabs(np.linalg.eig(A)[0]).max()


# sub question 1
def sub1():
    print()
    A, f = laplacian(10, lambda x, y: 1)
    A = -A
    f = -f
    print((np.linalg.eig(A)[0] >= 0).all())
    print(rho(A))

# sub question 2
def sub2():
    print()
    A, f = laplacian(20, lambda x, y: 1)
    A = -A
    f = -f
    print((np.linalg.eig(A)[0] >= 0).all())
    print(rho(A))

# sub question 3
def sub3():
    print()
    A_10, f_10 = laplacian(10, lambda x, y: 1)
    A_20, f_20 = laplacian(20, lambda x, y: 1)
    A_10 = -A_10
    f_10 = -f_10
    A_20 = -A_20
    f_20 = -f_20
    space = np.linspace(0.25 * 0.999, 2 / max(rho(A_10), rho(A_20)), num=20, endpoint=True)
    for alpha in space:
        u_10 = f_10.copy()
        u_20 = f_20.copy()
        cnt = 0
        while (rres(A_10, f_10, u_10) > 1e-5 or \
               rres(A_20, f_20, u_20) > 1e-5) and \
              cnt < 5000:
            u_10 = u_10 + alpha * (f_10 - A_10 @ u_10)
            u_20 = u_20 + alpha * (f_20 - A_20 @ u_20)
            cnt += 1
        print("{:4d}: {}".format(cnt, alpha))

# run sub questions
sub1()
sub2()
sub3()
```