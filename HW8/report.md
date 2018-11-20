# Homework 8

- Jianfei Gao
- 0029986102
- **Problem 0**: I did not discuss with anyone except those discussions on Piazza. All text and code which are necessary are included.

##  Problem 1

1. Let's denote the summation of two Float64 numbers as $\oplus$.

   We need to show that
   $$
   f_{64}(\underline{x}) = \left( \cdots \left( \left( x_1 \oplus x_2 \right) \oplus x_3\right) \cdots \right) \oplus x_n = \left( \cdots \left( \left( \hat{x}_1 + \hat{x}_2 \right) + \hat{x}_3\right) \cdots \right) + \hat{x}_n = f(\hat{\underline{x}})
   $$
   where $\|\hat{\underline{x}} - \underline{x} \| \leq C_n \epsilon$ from some $\underline{x}$.

   Denote $\left( \cdots \left( \left( x_1 \oplus x_2 \right) \oplus x_3\right) \cdots \right) \oplus x_n = \hat{s}_n$.

   We can get such a $\underline{x}$ recursively.
   $$
   \hat{x}_i = \begin{cases}
   x_1 & i = 1 \\
   \hat{s}_{i - 1} \oplus x_i - \hat{s}_{i - 1} & \text{o.w.} \\
   \end{cases}
   $$
   Then, we will recursively have
   $$
   \hat{s}_i = \begin{cases}
   x_1 = \hat{x}_1 = \sum\limits_{k = 1}^{i}{\hat{x}_k} & i = 1 \\
   \hat{s}_{i - 1} \oplus x_i = \hat{s}_{i - 1} + \hat{x}_i = \sum\limits_{k = 1}^{i}{\hat{x}_k} & \text{o.w.} \\
   \end{cases}
   $$

   By this construction, we can always have $\hat{\underline{x}}$ that  $\hat{s}_n = f_{64}(\underline{x}) = \sum\limits_{i = 1}^{n}{\hat{x}_i} = f(\hat{\underline{x}})$.

   Let's show that this $\underline{x}$ will have $\|\hat{\underline{x}} - \underline{x}\|_{1} \leq C_n\epsilon$.

   On the class, we already show that $a \oplus b = (a + b) (1 + \sigma)$ where $|\sigma| \leq \epsilon$.

   So, for $\hat{x}_i = \hat{s}_{i - 1} \oplus x_i - \hat{s}_{i - 1}$, we will have
   $$
   \begin{aligned}
   \hat{x}_i &= (\hat{s}_{i - 1} + x_i) (1 + \sigma_i) - \hat{s}_{i - 1} \\
   \hat{x}_i &= \hat{s}_{i - 1} + x_i + (\hat{s}_{i - 1} + x_i) \sigma_i - \hat{s}_{i - 1} \\
   \hat{x}_i - x_i &= (\hat{s}_{i - 1} + x_i) \sigma_i
   \end{aligned}
   $$
   where $|\sigma_i| \leq \epsilon$.

   Then,
   $$
   \begin{aligned}
   \|\hat{\underline{x}} - \underline{x}\|_1 &= \sum\limits_{i = 1}^{n}{\left|\hat{x}_i - x_i\right|} \\
   &= \sum\limits_{i = 1}^{n}{\left| (\hat{s}_{i - 1} + x_i) \sigma_i \right|} \\
   &= \sum\limits_{i = 1}^{n}{\left| (\sum\limits_{k = 1}^{i - 1}{\hat{x}_i} + x_i) \right| \left| \sigma_i \right|} \\
   &\leq \sum\limits_{i = 1}^{n}{\left(\sum\limits_{k = 1}^{i - 1}{\left| \hat{x}_i \right|} + \left| x_i \right| \right) \left| \sigma_i \right|} \\
   &\leq \sum\limits_{i = 1}^{n}{\left(\sum\limits_{k = 1}^{n}{\left| \hat{x}_i \right|} + \left| x_i \right| \right) \epsilon} \\
   &= \left( n \left\|\hat{\underline{x}}\right\|_1  + \|\underline{x}\|_1 \right) \epsilon \\
   &\leq \left( n \left\|\underline{x}\right\|_1 + n \|\hat{\underline{x}} - \underline{x}\|_1 \right) + \|\underline{x}\|_1 \epsilon \\
   \end{aligned}
   $$
   So,
   $$
   \begin{aligned}
   \frac{\|\hat{\underline{x}} - \underline{x}\|_1}{\| \hat{\underline{x}} \|_1} &\leq  \frac{(n + 1)\epsilon}{1 - n\epsilon} = C_n \epsilon \\
   \end{aligned}
   $$
   where $C_n = \frac{n + 1}{1 - n\epsilon}$ and $\epsilon$ is a tiny constant which is round-off of Float64.

2. Just sort $a, b, c$ into ascent order which implies summing by the following way: $s = a + b + c$ s.t. $a \leq b \leq c$.

3. Sort elements of $\underline{x}$ to get a permutation $\underline{x}^*$ that $x_1^* \leq x_2^* \leq \cdots \leq x_n^*$.

   I use
   $$
   \begin{aligned}
   x_{1} &= 10^{16} \\
   x_{2} &= 1 \\
   &\vdots \\
   x_{10001} &= 1 \\
   \end{aligned}
   $$
   with Python.

   To sum by the raw $\underline{x}$, Python will give **10000000000000000.000000** as output which is wrong.

   To sum by the permutation $\underline{x}^*$, Python will give **10000000000010000.000000** as output which is the true result.

4. Kahan algorithm will also give true result **10000000000010000.000000**.

## Problem 2

I compare **bisection method** with **quadratic method** and **citardauq method**.

For each method, a calibration process is applied:

> For two roots $r_1$ and $r_2$ returned by an algorithm, compute the **residuals $|f(r_i)|$** for each $r_i$. Select the one that gives minimum residual. Suppose it is $r_1$, then we have another $\hat{r}_2 = \frac{c}{ar_1}$. Compare the residuals of $r_2$ and $\hat{r}_2$, and select the one with least residual as new root $r_2$.

The comparison criterion is:

> The one whose largest residual of two roots is smaller will be better.
> If the largest residuals are the same, then the one whose sum of two residuals is smaller will be better.

I compare those three methods on following cases.

| $a$  | $b$  | $c$  | Best Method |
| :--: | :--: | :--: | :-: |
| $1$ | $3$ | $2$ | Bisection |
| $1$ | $3$ | $\frac{9}{4} - 10^{-20}$ | Bisection |
| $10^{-20}$ | $3$ | $2$ | Bisection |
| $10^{-20}$ | $3$ | $2 \times 10^{20}$ | Bisection |
| $10^{-20}$ | $3$ | $2 \times 10^{-20}$ | Citardauq |
| $10^{-20}$ | $3 \times 10^{-20}$ | $2 \times 10^{-20}$ | Bisection |

My observation from the outputs of my results is:

- Bisection method may not give as precise roots as other methods, but it will give roots that make $f(x)$ as close to 0 as possible which takes float round-off into consideration.

## Problem 3

1. Suppose $s_n = \sum\limits_{i = 1}^{n}{x_i y_i}$ is the true value, and $\hat{s}_n = \text{float}\left(\sum\limits_{i = 1}^{n}{x_i y_i}\right)$ is the true given by float operations. Denote the float round-off as $\epsilon$. Denote multiplication round-off of each step as $\pi_i$ and summation round-off of each step as $\sigma_{i}$.
   $$
   \begin{aligned}
   s_1 &= x_1 y_1 \\
   \hat{s}_1 &= x_1 y_1 (1 + \pi_1) \\
   |e_1| &= |x_1| |y_1| \pi_1 \\
   &\leq |x_1| |y_1| \epsilon
   \end{aligned}
   $$

   Suppose
   $$
   |e_k| \leq 2k \sum\limits_{i = 1}^{k}{|x_i| |y_i| \epsilon} + O(k \epsilon^2)
   $$

   Then, we can iterate.
   $$
   \begin{aligned}
   |e_{k + 1}| &= |[\hat{s}_{k} + x_{k + 1} y_{k + 1} (1 + \pi_{k + 1})] (1 + \sigma_{k + 1}) - s_{k} - x_{k + 1} y_{k + 1}| \\
   &= |[s_{k} + e_{k} + x_{k + 1} y_{k + 1} (1 + \pi_{k + 1})] (1 + \sigma_{k + 1}) - s_{k} - x_{k + 1} y_{k + 1}| \\
   &= |[e_{k} + s_{k + 1} + x_{k + 1} y_{k + 1} \pi_{k + 1}] (1 + \sigma_{k + 1}) - s_{k + 1}| \\
   &= |(e_{k} + x_{k + 1} y_{k + 1} \pi_{k + 1}) (1 + \sigma_{k + 1}) + s_{k + 1} \sigma_{k + 1}| \\
   &= |(e_{k} + x_{k + 1} y_{k + 1} \pi_{k + 1}) \sigma_{k + 1} + e_{k} + x_{k + 1} y_{k + 1} \pi_{k + 1} + s_{k + 1} \sigma_{k + 1}| \\
   &\leq |(e_{k} + x_{k + 1} y_{k + 1} \pi_{k + 1}) \sigma_{k + 1}| + |e_{k}| + |x_{k + 1}| |y_{k + 1}| \pi_{k + 1} + |s_{k + 1}| \sigma_{k + 1} \\
   &\leq |(e_{k} + x_{k + 1} y_{k + 1} \pi_{k + 1}) \sigma_{k + 1}| + |e_{k}| + |x_{k + 1}| |y_{k + 1}| \pi_{k + 1} + \sum\limits_{i = 1}^{k + 1}{|x_i| |y_i|} \sigma_{k + 1} \\
   &\leq |e_{k}| + |x_{k + 1}| |y_{k + 1}| \epsilon + \sum\limits_{i = 1}^{k + 1}{|x_i| |y_i|} \epsilon + |e_{k} + x_{k + 1} y_{k + 1} \pi_{k + 1}| \epsilon \\
   &\leq |e_{k}| (1 + \epsilon) + 2 \sum\limits_{i = 1}^{k + 1}{|x_i| |y_i|} \epsilon + O(\epsilon^2) \\
   &\leq |e_{k}| + |x_{k + 1}| |y_{k + 1}| \epsilon + \sum\limits_{i = 1}^{k + 1}{|x_i| |y_i|} \epsilon + |e_{k} + x_{k + 1} y_{k + 1} \pi_{k + 1}| \epsilon \\
   &\leq \left( 2k \sum\limits_{i = 1}^{k}{|x_i| |y_i| \epsilon} + O(k \epsilon^2) \right) (1 + \epsilon) + 2 \sum\limits_{i = 1}^{k + 1}{|x_i| |y_i|} \epsilon + O(\epsilon^2) \\
   &\leq 2k \sum\limits_{i = 1}^{k}{|x_i| |y_i| \epsilon} + O(k \epsilon^2) + O(k\epsilon^2) + 2 \sum\limits_{i = 1}^{k + 1}{|x_i| |y_i|} \epsilon + O(\epsilon^2) \\
   &\leq 2(k + 1) \sum\limits_{i = 1}^{k}{|x_i| |y_i| \epsilon} + O((k + 1) \epsilon^2) \\
   \end{aligned}
   $$

   Thus, we will have
   $$
   \begin{aligned}
   |e_n| &\leq 2n \epsilon |\underline{x}|^\text{T} \underline{y} + O(n \epsilon^2) \\
   \frac{|e_n|}{|s_n|} &\leq \frac{n + O(n)}{2} \epsilon = \frac{1}{C_n} \epsilon \\
   \end{aligned}
   $$

   This means that as long as $\hat{x}_i = x_i (1 + \pi_i)$ where $|\pi_i| \leq C_n \epsilon$ which means $\frac{|\hat{x}_i - x_i|}{|x_i|} \leq C_n \epsilon$, we can always have $\frac{|e_n|}{|s_n|} \leq \epsilon$ which means $\underline{x}^\text{T} \underline{y} = \hat{\underline{x}}^\text{T} \underline{y}$. So, dot product is backward stable.

2. From the dot product prove of previous sub question, moving the error part from $\underline{x}$ to $\underline{y}$, we can get

   > If we have $\hat{y}_i = y_i (1 + \pi_i)$ where $|\pi_i| \leq C_n \epsilon$ which means $\frac{|\hat{y}_i - y_i|}{|y_i|} \leq C_n \epsilon$, $\text{float}\left(\underline{a}^\text{T} \underline{y}\right) = \underline{a}^\text{T} \hat{\underline{y}}$

   Based on this conclusion, we can have following view.

   Let
   $$
   A\underline{x} = \begin{bmatrix}
   \underline{a}_1^\text{T} \\
   \vdots \\
   \underline{a}_n^\text{T} \\
   \end{bmatrix} \underline{x}
   = \begin{bmatrix}
   \underline{a}_1^\text{T} \underline{x} \\
   \vdots \\
   \underline{a}_n^\text{T} \underline{x} \\
   \end{bmatrix}
   $$

   We also know that for $\hat{x}_i = x_i (1 + \pi_i)$ where $|\pi_i| \leq C_n \epsilon$ which means $\frac{|\hat{x}_i - x_i|}{|x_i|} \leq C_n \epsilon$, $\text{float}\left(\underline{a}_i^\text{T} \underline{x}\right) = \underline{a}_i^\text{T} \hat{\underline{x}}$ for any $i$.

   Thus
   $$
   \text{float}\left(A\underline{x}\right)
   = \text{float}\left(\begin{bmatrix}
   \underline{a}_1^\text{T} \underline{x} \\
   \vdots \\
   \underline{a}_n^\text{T} \underline{x} \\
   \end{bmatrix} \right)
   = \begin{bmatrix}
   \text{float}\left( \underline{a}_1^\text{T} \underline{x} \right) \\
   \vdots \\
   \text{float}\left( \underline{a}_n^\text{T} \underline{x} \right) \\
   \end{bmatrix}
   = \begin{bmatrix}
   \underline{a}_1^\text{T} \hat{\underline{x}} \\
   \vdots \\
   \underline{a}_n^\text{T} \hat{\underline{x}} \\
   \end{bmatrix}
   = A\hat{\underline{x}}
   $$



## Problem 4

BigFloat is more accurate.

| Float16 | BigFloat                                                     |
| ------- | ------------------------------------------------------------ |
| 33.47   | 3.366804940823772789066808753275966945478297585658036524204765786756561844815972e+01 |

The codes are located at Appendix 3

## Appendix 1

```python
def raw_sum(x):
    return sum(x)

def perm_sum(x):
    return sum(sorted(x))

def kahan_sum(x):
    s, c = 0, 0
    y, t = None, None
    for itr in x:
        y = itr - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


# sub question 4
def p4():
    x = [1e16] + [1] * int(1e4)
    print("Raw    : {:f}".format(raw_sum(x)))
    print("Permute: {:f}".format(perm_sum(x)))
    print("Kahan  : {:f}".format(kahan_sum(x)))

# run sub questions
p4()
```

## Appendix 2

```python
import math


def f(a, b, c, x):
    if x is None:
        return None
    else:
        return a * (x ** 2) + b * x + c

def sign(v):
    if v > 0:
        return 1
    elif v < 0:
        return -1
    else:
        return 0

def bisect(a, b, c):
    t = b ** 2 - 4 * a * c
    if t < 0:
        return None, None
    else:
        t = math.sqrt(t)
    l0 = (-b - max(2 * t, 1)) / (2 * a)
    m0 = -b / (2 * a)
    r0 = (-b + max(2 * t, 1)) / (2 * a)
    if sign(f(a, b, c, l0)) != sign(f(a, b, c, m0)):
        x1 = l0
        x2 = m0
        m = (x1 + x2) / 2
        while (x1 < m) and (m < x2):
            if sign(f(a, b, c, m)) != sign(f(a, b, c, x2)):
                x1 = m
            else:
                x2 = m
            m = (x1 + x2) / 2
        r1 = m
    else:
        r1 = None
    if sign(f(a, b, c, m0)) != sign(f(a, b, c, r0)):
        x1 = m0
        x2 = r0
        m = (x1 + x2) / 2
        while (x1 < m) and (m < x2):
            if sign(f(a, b, c, x1)) != sign(f(a, b, c, m)):
                x2 = m
            else:
                x1 = m
            m = (x1 + x2) / 2
        r2 = m
    else:
        r2 = None
    return r1, r2

def quadratic(a, b, c):
    t = b ** 2 - 4 * a * c
    if t < 0:
        return None, None
    else:
        t = math.sqrt(t)
    r1 = (-b - t) / (2 * a)
    r2 = (-b + t) / (2 * a)
    return r1, r2

def citardauq(a, b, c):
    t = b ** 2 - 4 * a * c
    if t < 0:
        return None, None
    else:
        t = math.sqrt(t)
    if -b + t == 0:
        r1 = None
    else:
        r1 = (2 * c) / (-b + t)
    if -b - t == 0:
        r2 = None
    else: 
        r2 = (2 * c) / (-b - t)
    return r1, r2

def fix(a, b, c, r1, r2):
    e1, e2 = f(a, b, c, r1), f(a, b, c, r2)
    if e1 is None and e2 is None:
        pass
    elif e1 is None or math.fabs(e1) > math.fabs(e2):
        e3 = f(a, b, c, c / a / r2)
        if e1 is None or math.fabs(e3) < math.fabs(e1):
            r1 = c / a / r2
        else:
            pass
    elif e2 is None or math.fabs(e2) > math.fabs(e1):
        e3 = f(a, b, c, c / a / r1)
        if e2 is None or math.fabs(e3) < math.fabs(e2):
            r2 = c / a / r1
        else:
            pass
    else:
        pass
    return r1, r2


# sub question 1
def q1(do_fix=True):
    print()
    print('-----' * 22)
    for a, b, c in (
        (1, 3, 2), (1, 3, 9 / 4 - 1e-20),
        (1e-20, 3, 2), (1e-20, 3, 2e20),
        (1e-20, 3, 2e-20), (1e-20, 3e-20, 2e-20)):
        r11, r12 = bisect(a, b, c)
        if do_fix:
            r11, r12 = fix(a, b, c, r11, r12)
        else:
            pass
        e11, e12 = f(a, b, c, r11), f(a, b, c, r12)
        r21, r22 = quadratic(a, b, c)
        if do_fix:
            r21, r22 = fix(a, b, c, r21, r22)
        else:
            pass
        e21, e22 = f(a, b, c, r21), f(a, b, c, r22)
        r31, r32 = citardauq(a, b, c)
        if do_fix:
            r31, r32 = fix(a, b, c, r31, r32)
        else:
            pass
        e31, e32 = f(a, b, c, r31), f(a, b, c, r32)
        s1 = "Bisection: {:20s} ({:20s}), {:20s} ({:20s})".format(
                'NaN' if e11 is None else ('%.16f' % math.fabs(e11))[0:20],
                'NaN' if r11 is None else ('%.16f' % math.fabs(r11))[0:20],
                'NaN' if e12 is None else ('%.16f' % math.fabs(e12))[0:20],
                'NaN' if r12 is None else ('%.16f' % math.fabs(r12))[0:20])
        s2 = "Quadratic: {:20s} ({:20s}), {:20s} ({:20s})".format(
                'NaN' if e21 is None else ('%.16f' % math.fabs(e21))[0:20],
                'NaN' if r21 is None else ('%.16f' % math.fabs(r21))[0:20],
                'NaN' if e22 is None else ('%.16f' % math.fabs(e22))[0:20],
                'NaN' if r22 is None else ('%.16f' % math.fabs(r22))[0:20])
        s3 = "Citardauq: {:20s} ({:20s}), {:20s} ({:20s})".format(
                'NaN' if e31 is None else ('%.16f' % math.fabs(e31))[0:20],
                'NaN' if r31 is None else ('%.16f' % math.fabs(r31))[0:20],
                'NaN' if e32 is None else ('%.16f' % math.fabs(e32))[0:20],
                'NaN' if r32 is None else ('%.16f' % math.fabs(r32))[0:20])
        res = [
            (max(math.fabs(e11), math.fabs(e12)), math.fabs(e11) + math.fabs(e12), s1),
            (max(math.fabs(e21), math.fabs(e22)), math.fabs(e21) + math.fabs(e22), s2),
            (max(math.fabs(e31), math.fabs(e32)), math.fabs(e31) + math.fabs(e32), s3)]
        for i, (_, _, s) in enumerate(sorted(res, key=lambda x: (x[0], x[1]))):
            print("{} {}".format(i, s))
        print('-----' * 22)

# run sub questions
q1(True)
```

## Appendix 3

```julia
using DelimitedFiles
using LinearAlgebra

function analyze(dtype, epsilon)
    candyland_matrix = readdlm("candyland-matrix.csv", ',')
    A = zeros(dtype, 140, 140)
    for i = 1 : size(candyland_matrix, 1)
        row = Int(candyland_matrix[i, 1])
        col = Int(candyland_matrix[i, 2])
        val = candyland_matrix[i, 3]
        A[row, col] = val
    end
    # b = ones(dtype, 140, 1)
    # b[134, 1] = 0
    # println((inv(I - transpose(A)) * b)[140, 1])
    b = zeros(dtype, 140, 1)
    b[140, 1] = 1
    k = 1
    p = A * b
    S = k * p
    while true
        k += 1
        p = A * p
        if k * norm(p) < epsilon
            break
        else
            S = S + k * p
        end
    end
    println(S[134, 1])
end

analyze(Float16, 1e-4)
analyze(BigFloat, 1e-16)
```
