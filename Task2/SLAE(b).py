import numpy as np
import matplotlib.pyplot as plt


def m_0k_plus(k, diff):
    return (Q_C ** k) / (1 - Q_C) * diff


def m_k_plus(diff):
    return Q_C / (1 - Q_C) * diff


def m_k_minus(diff):
    return 1 / (1 + Q_C) * diff


TAU = 0.6
Q_C = .662
N = 200
EXACT = np.ones(N)
N_ITER = 50

a = []
temp = []
for i in range(1, N + 1):
    for j in range(1, N + 1):
        temp.append(i if i == j else 1 / (i + j))
    a.append(temp)
    temp = []
a = np.matrix(a)
b = np.diag(np.diagonal(a))
b = np.linalg.inv(b)
L_Matrix = np.identity(N) - TAU * np.matmul(b, a)
Q_C = np.linalg.norm(L_Matrix, 1)
print(Q_C)


f = a.dot(EXACT).reshape((N, 1))
B_Vector = TAU * b.dot(f)


x = np.zeros(N)
ers = [np.linalg.norm(x - EXACT)]
mk0p = []
mkp = []
mkm = []
temp = np.linalg.norm(L_Matrix.dot(x) + B_Vector)

for i in range(N_ITER):
    x_new = L_Matrix.dot(x) + B_Vector
    ers.append(np.linalg.norm(x_new - EXACT))
    mk0p.append(m_0k_plus(i, temp))
    mkp.append(m_k_plus(np.linalg.norm(x_new - x)))
    mkm.append(m_k_minus(np.linalg.norm(x_new - x)))
    x = x_new


plt.plot(list(range(1, N_ITER + 1)), ers[1:], label='$||e_k||$', color='black')
plt.plot(list(range(1, N_ITER + 1)), mk0p, label='$M^{k,0}_-$', ls='-.', color='red')
plt.plot(list(range(1, N_ITER + 1)), mkp, label='$M^k_+$', ls='--', color='red')
plt.plot(list(range(1, N_ITER)), mkm[1:], label='$M_-^k$', ls='--', color='green')
plt.yscale("log")
plt.xlabel("k")
plt.legend()
plt.grid()
plt.show()