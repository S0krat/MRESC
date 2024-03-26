import numpy as np


def zeta(s, epsilon):
    result = 0.
    k = 1
    while (t := 1 / (k ** s)) > epsilon:
        result += t
        k += 1
    return result


def zeta_hat(s, epsilon):
    result = 0.
    k = 0
    while (t := 1 / ((2 * k + 1) ** s)) > epsilon:
        result += t
        k += 1
    return result


def h_func(i, epsilon):
    s_i = (i + 1) / i
    return (2 ** s_i) * zeta_hat(s_i, epsilon) / ((2 ** s_i - 1) * zeta(s_i, epsilon))


def int_func(ksi, i, j):
    m = i + j * j
    return np.sin(i * ksi) * np.sin(j * ksi) + (m * np.cos(m * ksi) + 2 * np.sin(m * ksi)) * np.exp(2 * ksi)


def integration(n, i, j):
    h = np.pi / (n - 1)
    xs = np.array([h * i for i in range(n)])
    ys = np.array([int_func(x, i, j) for x in xs])
    result = 0.
    for j in range(n - 1):
        result += (ys[j] + ys[j + 1]) / 2 * h
    return result


def gaussian(a_matrix, b):
    reshaped_b = b.reshape((len(b), 1))
    a_matrix = np.hstack((a_matrix, reshaped_b))
    for i in range(len(a_matrix)):
        for j in range(i + 1, len(a_matrix)):
            a_matrix[j] -= a_matrix[i] * a_matrix[j][i] / a_matrix[i][i]
    x = np.array([0] * len(b), dtype=float)
    i = len(a_matrix) - 1
    while i >= 0:
        x[i] = (a_matrix[i][-1] - sum(x * a_matrix[i][0:-1])) / a_matrix[i][i]
        i -= 1
    return x


def max_a_i(mat_dim, epsilon, n):
    a_mat = np.zeros((mat_dim, mat_dim))
    for i in range(mat_dim):
        for j in range(mat_dim):
            a_mat[i, j] = integration(n, i + 1, j + 1) * 2 / np.pi

    b_vec = np.zeros(mat_dim)
    for i in range(mat_dim):
        b_vec[i] = np.sqrt(h_func(i + 1, epsilon))

    return np.max(gaussian(a_mat, b_vec))


def z_8(mat_dim, epsilon, n, n_rk):
    h = 8 / (n_rk - 1)
    y = [1]
    z = -1  # max_a_i(mat_dim, epsilon, n)
    '''
    y''- 9y'- 10y = 0
        z' = 9z + 10y = U(-,y,z)
        y' = z = V(z)
    '''
    for i in range(n_rk):
        q_0 = 10 * y[-1] + 9 * z
        k_0 = z
        q_1 = 10 * (y[-1] + k_0 * h / 2) + 9 * (z + q_0 * h / 2)
        k_1 = z + q_0 * h / 2
        q_2 = 10 * (y[-1] + k_1 * h / 2) + 9 * (z + q_1 * h / 2)
        k_2 = z + q_1 * h / 2
        q_3 = 10 * (y[-1] + k_2 * h) + 9 * (z + q_2 * h)
        k_3 = z + q_2 * h
        z = z + h / 6 * (q_0 + 2 * q_1 + 2 * q_2 + q_3)
        y.append(y[-1] + h / 6 * (k_0 + 2 * k_1 + 2 * k_2 + k_3))

    import matplotlib.pyplot as plt
    plt.plot([i * h for i in range(n_rk + 1)], y)
    plt.grid()
    plt.yscale("log")
    plt.show()


z_8(10, 0.0001, 1000, 100)
