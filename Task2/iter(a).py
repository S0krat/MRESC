import numpy as np
import matplotlib.pyplot as plt

ALPHA_C = 0.9
P_C = 4
X_0 = 10
EXACT = (10 ** (1 / (P_C + 1)))
Q_C = 0.9
N_ITER = 30


def iter_pr(x_k):
    return ALPHA_C * x_k + 1 / (x_k ** P_C)


def m_0k_plus(k, diff):
    return (Q_C ** k) / (1 - Q_C) * diff


def m_k_plus(diff):
    return Q_C / (1 - Q_C) * diff


def m_k_minus(diff):
    return 1 / (1 + Q_C) * diff


def m_lk_plus(l, diff):
    return 1 / (1 - Q_C ** l) * diff


def m_lk_minus(l, diff):
    return 1 / (1 + Q_C ** l) * diff


x = [X_0]
ers = [np.abs(X_0 - EXACT)]
mk0p = []
mkp = []
mkm = []
mklp = []
mklm = []
for k in range(N_ITER):
    x.append(iter_pr(x[-1]))
    ers.append(np.abs(x[-1] - EXACT))
    mk0p.append(m_0k_plus(k, np.abs(x[0] - x[1])))
    mkp.append(m_k_plus(np.abs(x[k] - x[k + 1])))
    mkm.append(m_k_minus(np.abs(x[k] - x[k + 1])))
    if k > 0:
        mklp.append(m_lk_plus(2, np.abs(x[k - 1] - x[k + 1])))
        mklm.append(m_lk_minus(2, np.abs(x[k - 1] - x[k + 1])))


plt.plot(list(range(N_ITER + 1)), ers, label='$|e_k|$', color='black')
plt.plot(list(range(1, N_ITER + 1)), mk0p, label='$M^{k,0}_+$', ls='-.', color='red')
plt.plot(list(range(1, N_ITER + 1)), mkp, label='$M^k_+$', ls='--', color='red')
plt.plot(list(range(N_ITER)), mkm, label='$M_-^k$', ls='--', color='green')
plt.plot(list(range(N_ITER - 1)), mklp, label='$M^{k,2}_+$', ls=':', color='red')
plt.plot(list(range(N_ITER - 1)), mklm, label='$M^{k,2}_-$', ls=':', color='green')
plt.yscale("log")
plt.xlabel("k")
plt.title(f"$p={P_C},x_0={X_0}$")
plt.legend()
plt.grid()
plt.show()
