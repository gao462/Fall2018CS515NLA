# Homework 7

- Jianfei Gao
- 0029986102
- **Problem 0**: I did not discuss with anyone except those discussions on Piazza. All text and code which are necessary are included.

## Problem 1

1. Use 2-norm to compute condition number.
   $$
   \begin{aligned}
   \frac{\|J(x)\|\|x\|}{\|f(x)\|}
   &= \frac{\sqrt{\left( \frac{1}{N} \right)^2 \cdot N} \sqrt{\sum\limits_{i = 1}^{N}{x_i^2}}}{|\frac{1}{N} \sum\limits_{i = 1}^{N}{x_i}|} \\
   &= \frac{\sqrt{\frac{1}{N} \sum\limits_{i = 1}^{N}{x_i^2}}}{|\frac{1}{N}\sum\limits_{i = 1}^{N}{x_i}|} \\
   &= \sqrt{N} \frac{\sqrt{\sum\limits_{i = 1}^{N}{x_i^2}}}{|\sum\limits_{i = 1}^{N}{x_i}|}
   \end{aligned}
   $$

   When $\sum{x_i}$ is far from 0, it is ill-conditioned. When  $\sum{x_i}$ is close to 0, it is ill-conditioned.

2. Use 2-norm to compute condition number.
   $$
   \begin{aligned}
   \frac{\|J(x)\|\|x\|}{\|f(x)\|}
   &= \frac{2\sqrt{\sum\limits_{i = 1}^{N}{(x_i - \bar{x})^2}} \sqrt{\sum\limits_{i = 1}^{N}{x_i^2}}}{\frac{1}{N - 1} \sum\limits_{i = 1}^{N}{(x_i - \bar{x})^2}} \\
   &= 2(N - 1) \sqrt{\frac{\sum\limits_{i = 1}^{N}{x_i^2}}{\sum\limits_{i = 1}^{N}{(x_i - \bar{x})^2}}} \\
   \end{aligned}
   $$

   When some $x_i$ are far from $\bar{x}$, it is well-conditioned. When all $x_i$ are close to $\bar{x}$, it is ill-conditioned.

3. Use 2-norm to compute condition number.
   $$
   \begin{aligned}
   \frac{\|J(x)\|\|x\|}{\|f(x)\|}
   &= \|A\| \frac{\|x\|}{\|Ax\|} \\
   \end{aligned}
   $$

   When some $\|Ax\|$ are far from 0, it is well-conditioned. When $\|Ax\|$ are close to 0, it is ill-conditioned.

4. Use 2-norm to compute condition number.
   $$
   \frac{\|\frac{\partial f(W^\text{T}x)}{\partial x}\| \|x\|}{\|f(W^\text{T}x)\|}
   $$

   The conditional number is hard to compute, but we can notice that it will obviously be ill-conditioned when $\|f(W^\text{T}x)\|$ is close to zero. This means that all $y_i$ will be close to zero, which means that all $w_i^\text{T}x$ will diverge to $-\infty$.

## Problem 2

$$
\begin{aligned}
X &= USV^\text{T} \\
X^\text{T}X &= VS^2V^\text{T} \\
XX^\text{T} &= US^2U^\text{T} \\
\end{aligned}
$$

So, $U$ will be a group of eigenvectors of $XX^\text{T}$, $V$ will be a group of eigenvectors of $X^\text{T}X$, and $S$ will be a diagonal matrix of square roots of eigenvalues of $X^\text{T}X$ or $XX^\text{T}$.

1. Compute eigenvectors of $XX^\text{T}$.
   $$
   \begin{aligned}
   XX^\text{T} &= \begin{bmatrix}
   9 & 0 \\
   0 & 0 \\
   \end{bmatrix} \\
   (9 - \lambda) (0 -\lambda) - 0 &= 0 \\
   \lambda_1 &= 9 \\
   u_1 &= \begin{bmatrix}
   1 \\
   0 \\
   \end{bmatrix} \\
   \lambda_2 &= 0 \\
   u_2 &= \begin{bmatrix}
   0 \\
   1 \\
   \end{bmatrix}
   \end{aligned}
   $$

   Compute eigenvectors of $X^\text{T}X$.
   $$
   \begin{aligned}
   XX^\text{T} &= \begin{bmatrix}
   0 & 0 \\
   0 & 9 \\
   \end{bmatrix} \\
   (0 - \lambda) (9 -\lambda) - 0 &= 0 \\
   \lambda_1 &= 9 \\
   v_1 &= \begin{bmatrix}
   0 \\
   1 \\
   \end{bmatrix} \\
   \lambda_2 &= 0 \\
   v_2 &= \begin{bmatrix}
   1 \\
   0 \\
   \end{bmatrix}
   \end{aligned}
   $$

   So, we have
   $$
   X = USV^\text{T} =
   \begin{bmatrix}
   1 & 0 \\
   0 & 1 \\
   \end{bmatrix}
   \begin{bmatrix}
   3 & 0 \\
   0 & 0 \\
   \end{bmatrix}
   \begin{bmatrix}
   0 & 1 \\
   1 & 0 \\
   \end{bmatrix}
   $$

2. Compute eigenvectors of $XX^\text{T}$.
   $$
   \begin{aligned}
   XX^\text{T} &= \begin{bmatrix}
   25 & 10 \\
   10 & 4 \\
   \end{bmatrix} \\
   (25 - \lambda) (4 -\lambda) - 100 &= 0 \\
   \lambda_1 &= 29 \\
   u_1 &= \frac{1}{\sqrt{29}} \begin{bmatrix}
   5 \\
   2 \\
   \end{bmatrix} \\
   \lambda_2 &= 0 \\
   u_2 &= \frac{1}{\sqrt{29}} \begin{bmatrix}
   2 \\
   -5 \\
   \end{bmatrix}
   \end{aligned}
   $$

   Compute eigenvectors of $X^\text{T}X$.
   $$
   \begin{aligned}
   XX^\text{T} &= \begin{bmatrix}
   29 & 0 \\
   0 & 0 \\
   \end{bmatrix} \\
   (0 - \lambda) (9 -\lambda) - 0 &= 0 \\
   \lambda_1 &= 29 \\
   v_1 &= \begin{bmatrix}
   1 \\
   0 \\
   \end{bmatrix} \\
   \lambda_2 &= 0 \\
   v_2 &= \begin{bmatrix}
   0 \\
   1 \\
   \end{bmatrix}
   \end{aligned}
   $$

   So, we have
   $$
   X = USV^\text{T} =
   \begin{bmatrix}
   \frac{5}{\sqrt{29}} & \frac{2}{\sqrt{29}} \\
   \frac{2}{\sqrt{29}} & \frac{-5}{\sqrt{29}} \\
   \end{bmatrix}
   \begin{bmatrix}
   \sqrt{29} & 0 \\
   0 & 0 \\
   \end{bmatrix}
   \begin{bmatrix}
   1 & 0 \\
   0 & 1 \\
   \end{bmatrix}
   $$

3. Compute eigenvectors of $XX^\text{T}$.
   $$
   \begin{aligned}
   XX^\text{T} &= \begin{bmatrix}
   50 & 20 & 0 \\
   20 & 8 & 0 \\
   0 & 0 & 0 \\
   \end{bmatrix} \\
   (50 - \lambda) (8 -\lambda)(0 - \lambda) - 20 \cdot 20 \lambda &= 0 \\
   \lambda_1 &= 58 \\
   u_1 &= \frac{1}{\sqrt{29}} \begin{bmatrix}
   5 \\
   2 \\
   0 \\
   \end{bmatrix} \\
   \lambda_2 &= 0 \\
   u_2 &= \frac{1}{\sqrt{29}} \begin{bmatrix}
   2 \\
   -5 \\
   0 \\
   \end{bmatrix} \\
   \lambda_3 &= 0 \\
   u_3 &= \begin{bmatrix}
   0 \\
   0 \\
   1 \\
   \end{bmatrix}
   \end{aligned}
   $$

   Compute eigenvectors of $X^\text{T}X$.
   $$
   \begin{aligned}
   XX^\text{T} &= \begin{bmatrix}
   29 & -29 \\
   -29 & 29 \\
   \end{bmatrix} \\
   (29 - \lambda) (29 -\lambda) - (-29) (-29) &= 0 \\
   \lambda_1 &= 58 \\
   v_1 &= \frac{1}{\sqrt{2}} \begin{bmatrix}
   1 \\
   -1 \\
   \end{bmatrix} \\
   \lambda_2 &= 0 \\
   v_2 &= \frac{1}{\sqrt{2}} \begin{bmatrix}
   1 \\
   1 \\
   \end{bmatrix}
   \end{aligned}
   $$

   So, we have
   $$
   X = USV^\text{T} =
   \begin{bmatrix}
   \frac{5}{\sqrt{29}} & \frac{2}{\sqrt{29}} & 0 \\
   \frac{2}{\sqrt{29}} & \frac{-5}{\sqrt{29}} & 0 \\
   0 & 0 & 1 \\
   \end{bmatrix}
   \begin{bmatrix}
   \sqrt{58} & 0 \\
   0 & 0 \\
   0 & 0 \\
   \end{bmatrix}
   \begin{bmatrix}
   \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
   \frac{-1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
   \end{bmatrix}
   $$

4. The result is direct from $X$.
   $$
   X = USV^\text{T} =
   \begin{bmatrix}
   1 & 0 \\
   0 & 1 \\
   \end{bmatrix}
   \begin{bmatrix}
   2 & 0 \\
   0 & -1 \\
   \end{bmatrix}
   \begin{bmatrix}
   1 & 0 \\
   0 & 1 \\
   \end{bmatrix}
   $$


## Problem 3

1. For a vector $\underline{f}$, just let $\gamma$ be the mean of all elements of $\underline{f}$.

2. Suppose the matrix to center is $X$ wit shape $n \times n$. We construct an all-one matrix $A$ with shape $n \times n$.
   $$
   X - \frac{1}{n} A X
   $$

3. The largest singular value is 29828.082683.

4. We get a blurring face.
   ![u1](C:\Users\gao46\Documents\Linux\Workplace\CS515NLA\HW7\u1.png)

## Problem 4

Based on the whole paper and conclusion of [[1]](https://www.researchgate.net/publication/256305714_Relations_between_condition_numbers_and_the_convergence_of_the_Jacobi_method_for_real_positive_definite_matrices), my guess is "the higher condition number, the lower convergence rate (speed)".

I also try to follow [[2]](https://www.researchgate.net/publication/220726426_The_influence_of_a_matrix_condition_number_on_iterative_methods'_convergence) to do some numerical experiments to verify this guess.

![p4](C:\Users\gao46\Documents\Linux\Workplace\CS515NLA\HW7\p4.png)

We can see that with condition number increasing (ignoring log relative residual error less than -10), the relative error will grow which means the rate (speed) of convergence being low.

## Appendix 1

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234567890)


# load raw file
img_file = open('Yale_64.csv', 'r')
lbl_file = open('Yale_64_ids.csv', 'r')
img_content = list(img_file.readlines())
lbl_content = list(lbl_file.readlines())
img_file.close()
lbl_file.close()
img_data = []
lbl_data = []
for img_line, lbl_line in zip(img_content, lbl_content):
    image = img_line.strip().split(',')
    image = np.array([int(itr) for itr in image])
    label = int(lbl_line.strip())
    img_data.append(image)
    lbl_data.append(label)
img_mx = np.array(img_data, dtype=float).T
lbl_mx = np.array(lbl_data, dtype=int).reshape(1, -1)


# sub question 2
def sub2():
    print()
    n = img_mx.shape[0]
    avg_mx = np.ones(shape=(n, n)) / n
    img_cmx = img_mx - avg_mx @ img_mx

# sub question 3
def sub3():
    print()
    img_cmx = img_mx - img_mx.mean(axis=0)
    u_mx, s, vt_mx = np.linalg.svd(img_cmx)
    print(s.max())

# sub question 4
def sub4():
    print()
    img_cmx = img_mx - img_mx.mean(axis=0)
    u_mx, s, vt_mx = np.linalg.svd(img_cmx)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(u_mx[0, :].reshape(64, 64).T, cmap='gray')
    fig.savefig('u1.png')
    plt.close(fig)

# run sub questions
sub2()
sub3()
sub4()
```

## Appendix 2

```python
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234567890)

def loss(x, A, b):
    return np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    
def gauss_seidel(A, b):
    x = b.copy()
    cnt = 0
    old_loss = loss(x, A, b)
    rate = None
    while old_loss > 1e-4 and cnt < 100:
        for i in range(len(x)):
            sigma = b[i, 0]
            for j in range(0, len(x)):
                if j == i:
                    continue
                else:
                    sigma -= A[i, j] * x[j, 0]
            x[i, 0] = sigma / A[i, i]
        cnt += 1
        new_loss = loss(x, A, b)
        rate = new_loss / old_loss
        old_loss = new_loss
    return loss(x, A, b)

results = []
for N in (2, 3, 4):
    b = np.random.normal(size=(N, 1))
    for i in range(1000):
        A = np.ones(shape=(N, N))
        for r in range(A.shape[0]):
            for c in range(r + 1, A.shape[1]):
                if r == c:
                    continue
                else:
                    pass
                val = (np.random.random() - 0.5) * 2 * np.exp(np.random.randint(-20, 2))
                A[r, c] = val
                A[c, r] = val
        cond, metric = np.linalg.cond(A), gauss_seidel(A, b)
        if np.all(np.linalg.eig(A)[0] > 0) and np.all(np.diag(A) == 1) and np.all(A.T == A):
            results.append((cond, metric))
        else:
            pass

results = sorted(results, key=lambda x: x[0])
results = np.array(results, dtype=float)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(np.log(results[:, 0]), np.log(results[:, 1]), marker='.')
ax.set_xlabel('Log Condition Number')
ax.set_ylabel('Log Relative Residual (100 Iteration)')
fig.savefig('p4.png')
```