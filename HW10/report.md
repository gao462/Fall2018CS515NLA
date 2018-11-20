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

## Appendix 1

```python
import re
import numpy as np
from scipy.sparse.linalg import cg as _cg
np.random.seed(1234567890)


def load():
    mx = np.zeros(shape=(100, 100))
    file = open('p1.txt')
    content = file.readlines()
    file.close()
    for i, line in enumerate(content):
        if len(line) == 0:
            break
        nums = re.split(r'\s+', line.strip())
        nums = [int(itr) for itr in nums]
        assert len(nums) == 100
        for j in range(len(nums)):
            mx[i, j] = nums[j]
    return mx

def spddiagm(n):
    mx = np.zeros(shape=(n, n))
    for i in range(n):
        mx[i, i] = 4
    for i in range(n - 1):
        mx[i, i + 1] = -2
        mx[i + 1, i] = -2
    return mx

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
    return residuals

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
    return residuals


A = spddiagm(100)
b = np.ones(shape=(100, 1))
print(minres(A, b, iter=25))
print(cg(A, b, iter=25))
```

