# Homework 5

- Jianfei Gao
- 0029986102
- **Problem 0**: I did not discuss with anyone except those discussions on Piazza. All text and code which are necessary are included.

## Problem 1

We can always divided each row $A_{i,:}$ and element $b_{i}$ by $A_{i,i}$, so that we can suppose that we are working with a matrix with $A_{i,i} = 1$ for all $i$ on solving $A \underline{x} = \underline{b}$.

Suppose
$$
\begin{aligned}
A &= \begin{bmatrix}
a_{1,1} & a_{1,2} & \cdots & a_{1, n} \\
a_{2,1} & a_{2,2} & \cdots & a_{2, n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n,1} & a_{n,2} & \cdots & a_{n, n} \\
\end{bmatrix} \\
&= \begin{bmatrix}
0 & a_{1,2} & \cdots & a_{1, n} \\
0 & 0 & \cdots & a_{2, n} \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0 \\
\end{bmatrix} +
\begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1 \\
\end{bmatrix} +
\begin{bmatrix}
0 & 0 & \cdots & 0 \\
a_{2,1} & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
a_{n,1} & a_{n,2} & \cdots & 0 \\
\end{bmatrix} \\
&= U + I + L. \\
\end{aligned}
$$

Then, the Jacobi iteration should be
$$
\begin{aligned}
\underline{x}^{(k + 1)} = \underline{b} - (U + L) \underline{x}^{(k)}
\end{aligned}
$$

Focus on eigenvalues $\lambda$ and determinant of $-(U + L)$
$$
\begin{aligned}
\text{det}(\lambda I - (-(U + L))) &= 0 \\
\text{det}(U + \lambda I + L) &= 0. \\
\end{aligned}
$$

**We know that the Jacobi iteration (or any iteration method) will obviously converge if and only if $\rho \left( -(U + L) \right) = \max{\{|\lambda|\}} < 1$**.

Let suppose that there is eigenvalue $\lambda$ s.t. $|\lambda| \geq 1$.

From Observation 2 of [URL](https://math.boisestate.edu/~calhoun/teaching/Math565_Fall2013/codes/SIREV95.pdf):

> If the matrix, $A$, is irreducible and diagonally dominant, then $A$ is nonsingular.

From Definition (9), (17) and Lemma of [URL](https://math.boisestate.edu/~calhoun/teaching/Math565_Fall2013/codes/SIREV95.pdf):

> $$
> \begin{aligned}
> A &= D - B - C \\
> A_J(\lambda) &= \lambda D - B - C \\
> \end{aligned}
> $$
> For each $\lambda$, with $|\lambda| \geq 1$, if $A$ satisfies any of following properties:
>
> 1. $A​$ is strictly diagonally dominant
> 2. $A$ is diagonally dominant
> 3. $A$ is irreducible
>
> then both $A_J(\lambda)$ and $A_G(\lambda)​$ satisfy the same properties.
>

So, **$U + \lambda I + L$ is nonsingular when $|\lambda| \geq 1$, which means $\text{det}(U + \lambda I + L) \neq 0$ which conflicts with $\lambda$ being an eigenvalue of $-(U + L)$**.

So, $\rho \left( -(U + L) \right) < 1$ which implies Jacobi iteration must converge.

*Indeed, this prove is nearly the same as [URL](https://math.boisestate.edu/~calhoun/teaching/Math565_Fall2013/codes/SIREV95.pdf)*.

## Problem 2

Jacobi method takes 170 iterations, and Gauss-Seidel method takes 150 iterations.

## Problem 3

1. **False**. The eigenvalues of
   $$
   \begin{bmatrix}
   0 & 1 \\
   -1 & 0 \\
   \end{bmatrix}
   $$

   are not real.

2. **True**. Suppose the eigenvalues of $A^\text{T}A$ are $\{\lambda_1, \cdots, \lambda_n\}$, and the eigenvalues of $A^\text{T}A + \gamma I$ are $\{\mu_1, \cdots, \mu_n\}$. It is obvious that $\lambda_i \geq 0$, thus $\mu_i = \lambda_i + \gamma > 0$. Then $\text{det}(A^\text{T}A + \gamma I) = \prod\limits_{i = 1}^{n}{\mu_i} > 0$. So we have unique solution.

3. **True**. Pick any $\alpha$ which is not an eigenvalue of the matrix $A$.
   $$
   \begin{aligned}
   A = (A - \alpha I) + \alpha I
   \end{aligned}
   $$
   
   Obviously, $\alpha I$ is non-singular.
   We can also know that $\text{det}(A - \alpha I) \neq 0$ which implies $(A - \alpha I)$ is non-singular.

4. **True**. Suppose any two pair of eigenvectors and eigenvalues $\underline{v},\underline{w}$ and $\lambda,\mu$.
   $$
   \begin{aligned}
   \lambda (\underline{v} \cdot \underline{w})
   &= (\lambda \underline{v}) \cdot \underline{w} = (A \underline{v}) \cdot \underline{w} \\
   &= \underline{v} \cdot (A^\text{T} \underline{w}) = \underline{v} \cdot (A \underline{w}) = \underline{v} \cdot (\mu \underline{w}) = \mu (\underline{v} \cdot \underline{w}) \\
   (\lambda - \mu) (\underline{v} \cdot \underline{w}) &= 0 \\
   \underline{v} \cdot \underline{w} &= 0 \\
   \end{aligned}
   $$

5. **True**. Since $A$ and $A^\text{T}$ have the same determinant, we will have
   $$
   \begin{aligned}
   \text{det} \left( \lambda I - A^\text{T} \right) = \text{det} \left( (\lambda I - A)^\text{T} \right) = \text{det}(\lambda I - A).
   \end{aligned}
   $$

   We know that eigenvalues of $A$ are equivalent to the solutions of $\text{det}(\lambda I - A) = 0$. Thus, $A$ and $A^\text{T}$ have the same eigenvalues.
   
## Problem 4

- Sub Question 1

  $$
  \begin{aligned}
  x - x_i &= r_i \\
  \|r_i\| &= \epsilon \\
  r_i^\text{T} r_i &= \epsilon^2 \\
  x_i &= x - r_i \\
  \left| \lambda - \lambda_i \right|
  &= \left| \lambda - x_i^\text{T} A x_i \right| \\
  &= \left| \lambda - (x - r_i)^\text{T} A (x - r_i) \right| \\
  &= \left| \lambda - \left( x^\text{T} - r_i^\text{T} \right) (A x - A r_i) \right| \\
  &= \left| \lambda - \left( x^\text{T} A x - 2 x^\text{T} A r_i + r_i^\text{T} A r_i \right) \right| \\
  &= \left| 2 x^\text{T} A r_i - r_i^\text{T} A r_i \right| \\
  &\leq 2 \left| x^\text{T} A r_i \right| + \left| r_i^\text{T} A r_i \right| \\
  \left| x^\text{T} A r_i \right| &\leq \left\| x^\text{T} A\right\| \| r_i \| \\
  &= \left\| x^\text{T} A\right\| \epsilon \\
  &\leq \| x \| \left\| A\right\| \epsilon \\
  &= \left\| A\right\| \epsilon \\
  \left| \frac{r_i^\text{T} A r_i} {r_i^\text{T} r_i} \right| &\leq \rho(A) \\
  \left| r_i^\text{T} A r_i \right| &\leq \rho(A) \epsilon^2 \\
  \left| \lambda - \lambda_i \right| &\leq 2 \left| x^\text{T} A r_i \right| + \left| r_i^\text{T} A r_i \right| \\
  &\leq 2 \| A \| \epsilon + \rho(A) \epsilon^2
  \end{aligned}
  $$

  **That's the only conclusion I got, and I can't bound it tigher to only $\epsilon^2$.**

- Sub Question 2

  $$
  \begin{aligned}
  \underline{x} &= A^{-1} A \underline{x} = \lambda A^{-1} \underline{x} \\
  A^{-1} \underline{x} &= \lambda^{-1} \underline{x} \\
  \end{aligned}
  $$






## Problem 5

Follow the prove of Problem 1, we can know that
$$
U + L = \begin{bmatrix}
0 & 1 & 0 & 0 \\
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 1 \\
0 & 0 & 1 & 0 \\
\end{bmatrix}
$$

whose $\rho(-(U + L)) = 1.618 > 1$. So, the Jacobi iteration will not converge.

## Problem 6

- Reference
  1. Theorm 10.1.2, Matrix Computations (3rd Edition), p512, [URL](http://web.mit.edu/ehliu/Public/sclark/Golub%20G.H.,%20Van%20Loan%20C.F.-%20Matrix%20Computations.pdf)
  2. Theorem (Householder-John), [URL](http://www.uta.edu/faculty/rcli/Teaching/math5371/Notes/split.pdf)

For this question, I prefer the first one because it operates directly over matrices of Gauss-Seidel, while the second one is a more general case which includes this question. 

## Appendix 1

```python
import numpy as np
np.random.seed(1234567890)


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
# | Iteration Methods
# +--------------------------------------------------


def loss(x, A, b):
    return np.linalg.norm(A @ x - b) / np.linalg.norm(b)

def jacobi(A, b):
    D = np.diag(A)
    R = A - np.diag(D)
    D_inv = np.diag(1 / D)
    x = b.copy()
    cnt = 0
    while loss(x, A, b) > 1e-4:
        x = D_inv @ (b - R @ x)
        cnt += 1
    return x, cnt

def gauss_seidel(A, b):
    x = b.copy()
    cnt = 0
    while loss(x, A, b) > 1e-4:
        for i in range(len(x)):
            sigma = b[i, 0]
            for j in range(0, len(x)):
                if j == i:
                    continue
                else:
                    sigma -= A[i, j] * x[j, 0]
            x[i, 0] = sigma / A[i, i]
        cnt += 1
    return x, cnt

# run problem
print()
A, b = candy_land()
x, cnt = jacobi(A, b)
print(cnt)
x, cnt = gauss_seidel(A, b)
print(cnt)
```