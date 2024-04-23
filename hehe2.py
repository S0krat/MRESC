import numpy as np
import matplotlib.pyplot as plt


LEFT = 0
RIGHT = 1


def exact(x):
    return np.cos(x) * (1 - x) - np.sin(x)


def f_func(x):
    return (np.cos(2 * x) - np.sin(x)) * (1 - x) - 3 / 2 * np.sin(2 * x) - 2 * np.cos(x)


def simpson(func, a, b):
    return (b - a) / 6 * (func(a) + 4 * func((b + a) / 2) + func(b))


def calculate_error(xss, ys):
    s = 0
    h = xss[1] - xss[0]
    for i in range(len(xss) - 1):
        s += simpson(lambda x: (exact(x) - (ys[i + 1] - ys[i]) / h) ** 2,
                     xss[i], xss[i + 1])

    return np.sqrt(s)


def calculate_error_by_parts(xss, ys):
    s = []
    h = xss[1] - xss[0]
    for i in range(len(xss) - 1):
        s.append(simpson(lambda x: (exact(x) - (ys[i + 1] - ys[i]) / h) ** 2, xss[i], xss[i + 1]))

    return s


def estimate_error(xss, ys):
    s = 0
    h = xss[1] - xss[0]
    for i in range(len(xss) - 1):
        simp = simpson(lambda x: (np.cos(x) * (ys[i + 1] - ys[i]) / h - f_func(x)) ** 2, xss[i], xss[i + 1])
        s += simp

    s *= (h / np.pi) ** 2

    return np.sqrt(s)


def estimate_error_by_parts(xss, ys):
    s = []
    h = xss[1] - xss[0]
    for i in range(len(xss) - 1):
        s.append(simpson(lambda x: (np.cos(x) * (ys[i + 1] - ys[i]) / h - f_func(x)) ** 2, xss[i], xss[i + 1]) * (h / np.pi) ** 2)

    return s


def tdma_solver(a, b, c, d):
    nf = len(d)  # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    return xc


def create_solution(n):
    h = (RIGHT - LEFT) / n
    nodes = [LEFT + i * h for i in range(n + 1)]
    a = np.zeros(n - 2)
    b = np.zeros(n - 1)
    c = np.zeros(n - 2)
    d = np.zeros(n - 1)

    b[0] = -(np.cos(nodes[0]) - np.cos(nodes[2]) + 2 * h) / (h ** 2)
    c[0] = (np.cos(nodes[1]) - np.cos(nodes[2]) + h) / (h ** 2)
    d[0] = f_func(nodes[0]) * h / 6 + f_func(nodes[1]) * 2 * h / 3 + f_func(nodes[2]) * h / 6

    for i in range(1, n - 2):
        a[i - 1] = (np.cos(nodes[i]) - np.cos(nodes[i + 1]) + h) / (h ** 2)
        b[i] = -(np.cos(nodes[i]) - np.cos(nodes[i + 2]) + 2 * h) / (h ** 2)
        c[i] = (np.cos(nodes[i + 1]) - np.cos(nodes[i + 2]) + h) / (h ** 2)
        d[i] = f_func(nodes[i]) * h / 6 + f_func(nodes[i + 1]) * 2 * h / 3 + f_func(nodes[i + 2]) * h / 6

    a[n - 3] = (np.cos(nodes[n - 2]) - np.cos(nodes[n - 1]) + h) / (h ** 2)
    b[n - 2] = -(np.cos(nodes[n - 2]) - np.cos(nodes[n]) + 2 * h) / (h ** 2)
    d[n - 2] = f_func(nodes[n - 2]) * h / 6 + f_func(nodes[n - 1]) * 2 * h / 3 + f_func(nodes[n]) * h / 6

    return tdma_solver(a, b, c, d)


def task1():
    NS = [2 ** i for i in range(4, 14)]
    errors = []
    errors_lol = []
    errors_lol_est = []
    errors_est = []
    for N in NS:
        # print(N)
        H = (RIGHT - LEFT) / N
        xs = [LEFT + i * H for i in range(N + 1)]
        ans = create_solution(N)
        ans = np.insert(ans, 0, 0)
        ans = np.append(ans, 0)
        ans_lol = ans + [0.0001 * (np.sin(7 * x) + 2 * x) for x in xs]
        errors_lol.append(calculate_error(xs, ans_lol))
        errors_lol_est.append(estimate_error(xs, ans_lol))
        errors.append(calculate_error(xs, ans))
        errors_est.append(estimate_error(xs, ans))

    plt.plot(NS, errors, label="Ошибка галёркинской апп.", color='black')
    plt.plot(NS, errors_lol, label="Ошибка негалёркинской апп.", color='red')
    plt.plot(NS, errors_est, label="Оценка ошибки галёр.", color='green', ls='--')
    plt.plot(NS, errors_lol_est, label="Оценка ошибки негалёр.", color='red', ls='--')
    plt.grid()
    plt.legend()
    plt.xlabel("n splits")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()


def task2(N):
    H = (RIGHT - LEFT) / N
    xs = [LEFT + i * H for i in range(N + 1)]
    ans = create_solution(N)
    ans = np.insert(ans, 0, 0)
    ans = np.append(ans, 0)
    ans_lol = ans + [0.0001 * (np.sin(7 * x) + 2 * x) for x in xs]
    error_pb = calculate_error_by_parts(xs, ans)
    error_pb_lol = calculate_error_by_parts(xs, ans_lol)
    error_est_pb = estimate_error_by_parts(xs, ans)
    error_est_pb_lol = estimate_error_by_parts(xs, ans_lol)

    xs = [(xs[i] + xs[i + 1]) / 2 for i in range(N)]
    plt.plot(xs, error_pb, label="Лок. ошибка галёр.", color='black')
    plt.plot(xs, error_pb_lol, label="Лок. ошибка негалёр.", color='red')
    plt.plot(xs, error_est_pb, label="Оценка лок. ошибки галёр.", color='green', ls='--')
    plt.plot(xs, error_est_pb_lol, label="Оценка лок. ошибки негалёр.", color='red', ls='--')
    plt.legend()
    plt.yscale("log")
    plt.title(f"N={N}")
    plt.grid()
    plt.show()


task2(10000)

