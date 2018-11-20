# Homework 6

- Jianfei Gao
- 0029986102
- **Problem 0**: I did not discuss with anyone except those discussions on Piazza. All text and code which are necessary are included.

## Problem 1

1. Since the swaping operation for pivoting LU does not related to FLOPs, so we just focus on FLOPs of common LU in an iteration.

   Suppose we are solving $Ax = \underline{b}$ for $n \times n$ matrix $A$ and $n \times 1$ vector $\underline{b}$.

   Based on [code from course page](https://www.cs.purdue.edu/homes/dgleich/cs515-2018/julia/lecture-12-pivoting.html), we have

   ```julia
   D = A[2:end,2:end]
   c = A[1,2:end]
   d = A[2:end,1]
   α = A[1,1]
   y = solve1(D-d*c'/α, b[2:end]-b[1]/α*d)
   γ = (b[1] - c'*y)/α
   ```
   whose corresponding formulas (transpose being applied on some variables for convenience) are
   $$
   \begin{aligned}
   A^{(n)} = A &= \begin{bmatrix}
   \alpha & \underline{c}^\text{T} \\
   \underline{d} & D \\
   \end{bmatrix} \\
   \underline{x} &= \begin{bmatrix}
   \gamma \\
   \underline{y}
   \end{bmatrix} \\
   \underline{b}^{(n)} = \underline{b} &= \begin{bmatrix}
   \beta \\
   \underline{f} \\
   \end{bmatrix} \\
   A^{(n - 1)} &= D - \frac{1}{\alpha} \left( \underline{d} \underline{c}^\text{T}    \right) = D - \underline{d} \left( \frac{1}{\alpha} \underline{c}^\text{T} \right) \\
   b^{(n - 1)} &= \underline{f} - \frac{\beta}{\alpha} \underline{d}\\
   \gamma &= \frac{1}{\alpha} \left( \beta - \underline{c}^\text{T} \underline{y} \right) = \frac{\beta}{\alpha} - \left( \frac{1}{\alpha} \underline{c}^\text{T} \right)    \underline{y} \\
   \end{aligned}
   $$

   Then, we count for FLOPs for those formulas:

   1. $\frac{1}{\alpha}$ takes 1 FLOP

   2. $\frac{1}{\alpha}\underline{c}^\text{T}$ take $n - 1$ FLOPs

   3. $D - \left( \frac{1}{\alpha} \underline{d} \right) \underline{c}^\text{T} = \left[ D_{i,j} - \underline{d}_{i} \left( \frac{1}{\alpha} \underline{c}^\text{T} \right)_j \right]_{i=1,j=1}^{n-1,n-1}$ takes $(n - 1)^2$ FLOPs

   4. $\frac{\beta}{\alpha}$ takes 1 FLOP

   5. $\underline{f} - \frac{\beta}{\alpha} \underline{d} = \left[ \underline{f}_{i} - \frac{\beta}{\alpha} \underline{d}_{i} \right]_{i=1}^{n - 1}$ takes $n - 1$ FLOPs

   6. $\frac{\beta}{\alpha} - \left( \frac{1}{\alpha} \underline{c}^\text{T} \right) \underline{y} = \left[ \frac{\beta}{\alpha} - \left( \frac{1}{\alpha} \underline{c}^\text{T} \right)_{i} \underline{y}_{i} \right]_{i=1}^{n-1}$ takes $n - 1$ FLOPs

   So, it totally takes $2 + 3(n - 1) + (n - 1)^2$ FLOPs. In the iteration, $n$ will decrease, so we replace it by a variable $i$, and $i$ will traverse from $n$ to 2. For    $i = 1$, we only need a division FLOP $\frac{b^{(n - 1)}}{A^{(n - 1)}}$.

   The summation will be
   $$
   \begin{aligned}
   \sum\limits_{i = 2}^{n}{\left[ 2 + 3(i - 1) + (i - 1)^2 \right]} + 1
   &= \sum\limits_{i = 1}^{n - 1}{\left[ 2 + 3i + i^2 \right]} + 1 \\
   &= 2(n - 1) + 3 \sum\limits_{i = 1}^{n - 1}{i} + \sum\limits_{i = 1}^{n - 1}{i^2} + 1    \\
   &= 2(n - 1) + 3 \frac{(n - 1)n}{2} + \frac{(n - 1)n(2n - 1)}{6} + 1 \\
   &= 2n - 2 + \frac{3}{2}n^2 - \frac{3}{2}n + \frac{1}{3} n^3 - \frac{1}{2}n^2 +    \frac{1}{6}n + 1 \\
   &= \frac{1}{3} n^3 + n^2 + \frac{2}{3} n - 1
   \end{aligned}
   $$

2. I run each operation for 1000000 times, and compute the average costs. Here are the results.

   | | Seconds | FLOPs |
   | :----: | ---- | ---- |
   | + | 0.0000008285 | 1.002177 |
   | - | 0.0000008267 | 1.000000 |
   | * | 0.0000008322 | 1.006653 |
   | / | 0.0000008343 | 1.009193 |
   | sqrt | 0.0000008494 | 1.027459 |
   | sin | 0.0000008431 | 1.019838 |
   | abs | 0.0000008450 | 1.022136 |

## Problem 2

1. Recursively solve $A^{-1}b$ where $A$ is $2^{k} \times 2^{k}$ matrix, and $b$ is $2^{k} \times 1$ matrix.

   $$
   \begin{bmatrix}
   A_1 & A_2 \\
   A_3 & A_4 \\
   \end{bmatrix}
   \begin{bmatrix}
   x_1 \\
   x_2 \\
   \end{bmatrix}
   = \begin{bmatrix}
   b_1 \\
   b_2 \\
   \end{bmatrix}
   $$

   Solve $x_1$ first.
   $$
   A_1 x_1 + A_2 x_2 = b_1 \\
   x_1 = A_1^{-1} (b_1 - A_2 x_2) \\
   $$

   Then, solve $x_2$ based on that.
   $$
   A_3 A_1^{-1} (b_1 - A_2 x_2) + A_4 x_2 = b_2 \\
   \left[ A_4 - A_3 \left( A_1^{-1} A_2 \right) \right] x_2 = b_2 - A_3 \left( A_1^{-1} b_1 \right) \\
   x_2 = \left[ A_4 - A_3 \left( A_1^{-1} A_2 \right) \right]^{-1} \left[ b_2 - A_3 \left( A_1^{-1} b_1 \right) \right] \\
   $$
   So, we only need to recursively solve $A_1^{-1} b_1$ and $A_1^{-1}A_2$ and $\left[ A_4 - A_3 \left( A_1^{-1} A_2 \right) \right]^{-1} \left[ b_2 - A_3 \left( A_1^{-1} b_1 \right) \right]$ to get $x_2$, then use all above results to get $x_1$.

2. Let's first focus on a single iteration.

   Suppose the FLOPs of an iteration is $f(a \times a, a \times b)$.

   - $A_1^{-1}b_1$ takes $f(\frac{a}{2} \times \frac{a}{2}, \frac{a}{2} \times b)$ FLOPs

   - $A_1^{-1}A_2$ takes $f(\frac{a}{2} \times \frac{a}{2}, \frac{a}{2} \times \frac{a}{2})$ FLOPs

   - $A_3 \left( A_1^{-1} A_2 \right)$ takes $\left(\frac{a}{2}\right)^3$ FLOPs

   - $A_4 - A_3 \left( A_1^{-1} A_2 \right)$ takes $\left(\frac{a}{2}\right)^2$ FLOPs

   - $A_3 \left( A_1^{-1} b_1 \right)$ takes $\left(\frac{a}{2}\right)^2 b$ FLOPs

   - $b_2 - A_3 \left( A_1^{-1} b_1 \right)$ takes $\frac{a}{2} b$ FLOPs

   - $\left[ A_4 - A_3 \left( A_1^{-1} A_2 \right) \right]^{-1} \left[ b_2 - A_3 \left( A_1^{-1} b_1 \right) \right]$ takes $f(\frac{a}{2} \times \frac{a}{2}, \frac{a}{2} \times b)$ FLOPs

   - $\left( A_1^{-1} A_2 \right) x_2$ takes $\left( \frac{a}{2} \right)^2 b$ FLOPs

   - $\left( A_1^{-1} b_1 \right) - \left( A_1^{-1} A_2 \right) x_2$ takes $\frac{a}{2} b$ FLOPs

   The iteration totally takes

   $$
   \left( \frac{a}{2} \right)^3 + \left( \frac{a}{2} \right)^2 + 2 \left( \frac{a}{2} \right)^2 b + 2 \left( \frac{a}{2} \right) b + f(\frac{a}{2} \times \frac{a}{2}, \frac{a}{2} \times \frac{a}{2}) + 2 f(\frac{a}{2} \times \frac{a}{2}, \frac{a}{2} \times b)
   $$

   For convenience, let's replace $a$ by even number $2a$.
   $$
   \begin{aligned}
   f(2a, b) &= a^3 + a^2 + 2a^2b + 2ab + f(a, a) + f(a, b) \\
   &= a \left( a^2 + a + f(a, a) \right) + 2 b \left( a^2 + a + f(a, b) \right) \\
   \end{aligned}
   $$





## Problem 3

1. See Appendix 2.

   ![image67](C:\Users\gao46\Documents\Linux\Workplace\CS515NLA\HW6\image67.png)

2. Suppose each image $M_{i}$ has been flatten into a vector $\left[ M_{i,1}, M_{i,2}, \cdots, M_{i,4096} \right]^\text{T}$, and we are minimizing loss of constructing image $T = M_t$.
   $$
   \min\limits_{\left[c_2, c_4, \cdots, c_{164}\right]^\text{T}}
   {\left\|
   \begin{bmatrix}
   M_{t,1} \\
   M_{t,2} \\
   \vdots \\
   M_{t,4096} \\
   \end{bmatrix}
   - \begin{bmatrix}
   M_{2,1} & M_{4,1} & \cdots & M_{164,1} \\
   M_{2,2} & M_{4,2} & \cdots & M_{164,2} \\
   \vdots & \vdots & \ddots & \vdots \\
   M_{2,4096} & M_{4,4096} & \cdots & M_{164,4096} \\
   \end{bmatrix}
   \begin{bmatrix}
   c_2 \\
   c_4 \\
   \vdots \\
   c_{164} \\
   \end{bmatrix} \right\|^2}
   $$

   The worst image is 95th image, and the best image is 77th image.

3. $\sigma$ can grow up to 191 before we won't recognize it as image 67 or person 7.

## Problem 4

1. They are different things
   - Householder
     $$
     \begin{aligned}
     H &= I - 2\frac{(x - \|x\| e_1)(x - \|x\| e_1)^\text{T}}{(x - \|x\| e_1)^\text{T}(x - \|x\| e_1)} \\
     &= I - 2 \frac{\begin{bmatrix}
     x_1 - \sqrt{x_1^2 + x_2^2} \\
     x_2 \\
     \end{bmatrix}
     \begin{bmatrix}
     x_1 - \sqrt{x_1^2 + x_2^2} & x_2 \\
     \end{bmatrix}}
     {2 \left( x_1^2 + x_2^2 - \sqrt{x_1^2 + x_2^2} \right)} \\
     &= \begin{bmatrix}
     1 - \frac{2x_1^2 + x_2^2 - 2\sqrt{x_1^2 + x_2^2}}{x_1^2 + x_2^2 - \sqrt{x_1^2 + x_2^2}} &
     1 - \frac{x_1 x_2 - x_2 \sqrt{x_1^2 + x_2^2}}{x_1^2 + x_2^2 - \sqrt{x_1^2 + x_2^2}} \\
     1 - \frac{x_1 x_2 - x_2 \sqrt{x_1^2 + x_2^2}}{x_1^2 + x_2^2 - \sqrt{x_1^2 + x_2^2}} &
     1 - \frac{x_2^2}{x_1^2 + x_2^2 - \sqrt{x_1^2 + x_2^2}} \\
     \end{bmatrix}
     \end{aligned}
     $$

   - Givens
     $$
     \begin{aligned}
     \begin{bmatrix}
     \frac{x_1}{\sqrt{x_1^2 + x_2^2}} & \frac{x_2}{\sqrt{x_1^2 + x_2^2}} \\
     \frac{-x_2}{\sqrt{x_1^2 + x_2^2}} & \frac{x_1}{\sqrt{x_1^2 + x_2^2}} \\
     \end{bmatrix}
     \begin{bmatrix}
     x_1 \\
     x_2 \\
     \end{bmatrix}
     = \begin{bmatrix}
     \sqrt{x_1^2 + x_2^2} \\
     0 \\
     \end{bmatrix}
     \end{aligned}
     $$
     $$
     G_1 = \begin{bmatrix}
     \frac{x_1}{\sqrt{x_1^2 + x_2^2}} & \frac{x_2}{\sqrt{x_1^2 + x_2^2}} \\
     \frac{-x_2}{\sqrt{x_1^2 + x_2^2}} & \frac{x_1}{\sqrt{x_1^2 + x_2^2}} \\
     \end{bmatrix}
     $$

2. They are different things
   - Householder
     $$
     \begin{aligned}
     H &= I - 2\frac{(x - \|x\| e_1)(x - \|x\| e_1)^\text{T}}{(x - \|x\| e_1)^\text{T}(x - \|x\| e_1)} \\
     &= I - 2 \frac{\begin{bmatrix}
     x_1 - \sqrt{x_1^2 + x_2^2 + x_3^2} \\
     x_2 \\
     x_3 \\
     \end{bmatrix}
     \begin{bmatrix}
     x_1 - \sqrt{x_1^2 + x_2^2 + x_3^2} & x_2 & x_3 \\
     \end{bmatrix}}
     {2 \left( x_1^2 + x_2^2 + x_3^2 - \sqrt{x_1^2 + x_2^2 + x_3^2} \right)} \\
     &= \begin{bmatrix}
     1 - \frac{2x_1^2 + x_2^2 + x_3^2 - 2 x_1 \sqrt{x_1^2 + x_2^2 + x_3^2}}{x_1^2 + x_2^2 + x_3^2 - \sqrt{x_1^2 + x_2^2 + x_3^2}} &
     1 - \frac{x_1 x_2 - x_2 \sqrt{x_1^2 + x_2^2 + x_3^2}}{x_1^2 + x_2^2 + x_3^2 - \sqrt{x_1^2 + x_2^2 + x_3^2}} &
     1 - \frac{x_1 x_3 - x_3 \sqrt{x_1^2 + x_2^2 + x_3^2}}{x_1^2 + x_2^2 + x_3^2 - \sqrt{x_1^2 + x_2^2 + x_3^2}} \\
     1 - \frac{x_1 x_2 - x_2 \sqrt{x_1^2 + x_2^2 + x_3^2}}{x_1^2 + x_2^2 + x_3^2 - \sqrt{x_1^2 + x_2^2 + x_3^2}} &
     1 - \frac{x_2^2}{x_1^2 + x_2^2 + x_3^2 - \sqrt{x_1^2 + x_2^2 + x_3^2}} &
     1 - \frac{x_2 x_3}{x_1^2 + x_2^2 + x_3^2 - \sqrt{x_1^2 + x_2^2 + x_3^2}} \\
     1 - \frac{x_1 x_3 - x_3 \sqrt{x_1^2 + x_2^2 + x_3^2}}{x_1^2 + x_2^2 + x_3^2 - \sqrt{x_1^2 + x_2^2 + x_3^2}} &
     1 - \frac{x_2 x_3}{x_1^2 + x_2^2 + x_3^2 - \sqrt{x_1^2 + x_2^2 + x_3^2}} &
     1 - \frac{x_3^2}{x_1^2 + x_2^2 + x_3^2 - \sqrt{x_1^2 + x_2^2 + x_3^2}} \\
     \end{bmatrix}
     \end{aligned}
     $$

   - Givens
     $$
     \begin{aligned}
     \begin{bmatrix}
     1 & 0 & 0 \\
     0 & \frac{x_2}{\sqrt{x_2^2 + x_3^2}} & \frac{x_3}{\sqrt{x_2^2 + x_3^2}} \\
     0 & \frac{-x_3}{\sqrt{x_2^2 + x_3^2}} & \frac{x_2}{\sqrt{x_2^2 + x_3^2}} \\
     \end{bmatrix}
     \begin{bmatrix}
     x_1 \\
     x_2 \\
     x_3 \\
     \end{bmatrix}
     &= \begin{bmatrix}
     x_1 \\
     \sqrt{x_2^2 + x_3^2} \\
     0 \\
     \end{bmatrix} \\
     
     \begin{bmatrix}
     \frac{x_1}{\sqrt{x_1^2 + x_2^2 + x_3^2}} & \frac{\sqrt{x_2^2 + x_3^2}}{\sqrt{x_1^2   + x_2^2 + x_3^2}} & 0 \\
     \frac{-\sqrt{x_2^2 + x_3^2}}{\sqrt{x_1^2 + x_2^2 + x_3^2}} & \frac{x_1}{\sqrt{x_1^2   + x_2^2 + x_3^2}} & 0 \\
     0 & 0 & 1 \\
     \end{bmatrix}
     \begin{bmatrix}
     x_1 \\
     \sqrt{x_2^2 + x_3^2} \\
     0 \\
     \end{bmatrix}
     &= \begin{bmatrix}
     \sqrt{x_1^2 + x_2^2 + x_3^2} \\
     0 \\
     0 \\
     \end{bmatrix} \\
     \end{aligned}
     $$
     $$
     \begin{aligned}
     G_2 G_1 &= \begin{bmatrix}
     \frac{x_1}{\sqrt{x_1^2 + x_2^2 + x_3^2}} & \frac{\sqrt{x_2^2 + x_3^2}}{\sqrt{x_1^2   + x_2^2 + x_3^2}} & 0 \\
     \frac{-\sqrt{x_2^2 + x_3^2}}{\sqrt{x_1^2 + x_2^2 + x_3^2}} & \frac{x_1}{\sqrt{x_1^2   + x_2^2 + x_3^2}} & 0 \\
     0 & 0 & 1 \\
     \end{bmatrix}
     \begin{bmatrix}
     1 & 0 & 0 \\
     0 & \frac{x_2}{\sqrt{x_2^2 + x_3^2}} & \frac{x_3}{\sqrt{x_2^2 + x_3^2}} \\
     0 & \frac{-x_3}{\sqrt{x_2^2 + x_3^2}} & \frac{x_2}{\sqrt{x_2^2 + x_3^2}} \\
     \end{bmatrix} \\
     &= \begin{bmatrix}
     \frac{x_1}{\sqrt{x_1^2 + x_2^2 + x_3^2}} & -\frac{-\sqrt{x_2^2 + x_3^2}}{\sqrt{x_1^2 + x_2^2 + x_3^2}} \frac{x_2}{\sqrt{x_2^2 + x_3^2}} & \frac{-\sqrt{x_2^2 + x_  3^2}}{\sqrt{x_1^2 + x_2^2 + x_3^2}} \frac{-x_3}{\sqrt{x_2^2 + x_3^2}} \\
     \frac{-\sqrt{x_2^2 + x_3^2}}{\sqrt{x_1^2 + x_2^2 + x_3^2}} & \frac{x_1}{\sqrt{x_1^2 + x_2^2 + x_3^2}} \frac{x_2}{\sqrt{x_2^2 + x_3^2}} & -\frac{x_1}{\sqrt{x_1^2 + x_  2^2 + x_3^2}} \frac{-x_3}{\sqrt{x_2^2 + x_3^2}} \\
     0 & \frac{-x_3}{\sqrt{x_2^2 + x_3^2}} & \frac{x_2}{\sqrt{x_2^2 + x_3^2}} \\
     \end{bmatrix} \\
     &= \begin{bmatrix}
     \frac{x_1}{\sqrt{x_1^2 + x_2^2 + x_3^2}} & \frac{x_2}{\sqrt{x_1^2 + x_2^2 + x_  3^2}} & \frac{x_3}{\sqrt{x_1^2 + x_2^2 + x_3^2}} \\
     -\frac{\sqrt{x_2^2 + x_3^2}}{\sqrt{x_1^2 + x_2^2 + x_3^2}} & \frac{x_1 x_2}{\sqrt{x_1^2 + x_2^2 + x_3^2} \sqrt{x_2^2 + x_3^2}} & \frac{x_1 x_3}{\sqrt{x_1^2 + x_2^2 +   x_3^2} \sqrt{x_2^2 + x_3^2}} \\
     0 & -\frac{x_3}{\sqrt{x_2^2 + x_3^2}} & \frac{x_2}{\sqrt{x_2^2 + x_3^2}} \\
     \end{bmatrix}
     \end{aligned}
     $$

3. My guess is "Householder method and Givens method can give two different solutions; Householder will give a symmetric matrix, but Givens will not."

4. To compute $\underline{v}$, we need
   $$
   \underline{v} = \underline{a} - \|\underline{a}\| e_1
   $$
   To compute $\|\underline{a}\|$, we need $n$ square operations, $n - 1$ sum operation and 1 square root operation, which totally are $2n$ FLOPs. To compute $\underline{v}$, we just need to subtract the first element which is 1 FLOP. So we need $2n + 1$ FLOPs to compute $\underline{v}$.

   To compute sine and cosine for a specific Givens matrix, we will have form
   $$
   \sin\theta_i = \frac{x_i}{\sqrt{x_i^2 + s_{i - 1}^2}} = \frac{x_i}{\sqrt{x_i^2 + \sum\limits_{j = 1}^{i - 1}{x_j^2}}}  \\
   \cos\theta_i = \frac{s_{i - 1}}{\sqrt{x_i^2 + s_{i - 1}^2}} = \frac{\sqrt{\sum\limits_{j = 1}^{i - 1}{x_j^2}}}{\sqrt{x_i^2 + \sum\limits_{j = 1}^{i - 1}{x_j^2}}} \\
   s_i^2 = x_i^2 + s_{i - 1}^2 = x_i^2 + \sum\limits_{j = 1}^{i - 1}{x_j^2} \\
   s_i = \sqrt{s_i^2}
   $$
   Suppose we save both $s_i$ and $s_i^2$ at every step. To compute $\sqrt{x_i^2 + s_{i - 1}^2}$, we need 1 square operation (2 for the first step), 1 sum operation and 1 square root operation. (In the meanwhile, we get $s_i$ and $s_i^2$.) To compute $\sin\theta_i$ and $\cos\theta_i$, we need 2 divide operations. So, we totally have 3 FLOPs in an iteration, and we will iterate $n - 1$ times, so we need $5(n - 1) + 1$ FLOPs to compute all sines and cosines.

## Appendix 1

```python
import time
import random
import math


N = 1000000

a = random.random() * 2 - 1
b = random.random() * 2 - 1
c = 0
s = 0
for i in range(N):
    timer = time.time()
    c = a + b
    s += (time.time() - timer)
print(" +   {:.10f}".format(s / N))

a = random.random() * 2 - 1
b = random.random() * 2 - 1
c = 0
s = 0
for i in range(N):
    timer = time.time()
    c = a - b
    s += (time.time() - timer)
print(" -   {:.10f}".format(s / N))

a = random.random() * 2 - 1
b = random.random() * 2 - 1
c = 0
s = 0
for i in range(N):
    timer = time.time()
    c = a * b
    s += (time.time() - timer)
print(" *   {:.10f}".format(s / N))

a = random.random() * 2 - 1
b = random.random() * 2 - 1
c = 0
s = 0
for i in range(N):
    timer = time.time()
    c = a / b
    s += (time.time() - timer)
print(" /   {:.10f}".format(s / N))

a = random.random()
c = 0
s = 0
sqrt = math.sqrt
for i in range(N):
    timer = time.time()
    c = sqrt(a)
    s += (time.time() - timer)
print("sqrt {:.10f}".format(s / N))

a = random.random() * 2 - 1
c = 0
s = 0
sin = math.sin
for i in range(N):
    timer = time.time()
    c = sin(a)
    s += (time.time() - timer)
print("sin  {:.10f}".format(s / N))

a = random.random() * 2 - 1
c = 0
s = 0
for i in range(N):
    timer = time.time()
    c = abs(a)
    s += (time.time() - timer)
print("abs  {:.10f}".format(s / N))
```

## Appendix 2

```python

```