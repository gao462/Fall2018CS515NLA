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