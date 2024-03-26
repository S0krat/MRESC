import numpy as np
import matplotlib.pyplot as plt

EXACT_SOLUTION = 0.2193839343949854
EPSILON = 1


def func(x: float) -> float:
    return np.exp(-x) / x


def integration_from_left(eps: float, n: int) -> float:
    h = (25 - eps) / (n - 1)
    xs = np.array([eps + h * i for i in range(n)])
    ys = np.array([func(x) for x in xs])
    print((ys[-1] + ys[-2]) / 2 * h)
    result = 0.
    for j in range(n - 1):
        result += (ys[j] + ys[j + 1]) / 2 * h
    return result


def integration_from_right(eps: float, n: int) -> float:
    h = (25 - eps) / (n - 1)
    xs = [eps + h * i for i in range(n)]
    ys = [func(x) for x in xs]
    result = 0.
    for j in range(n - 2, -1, -1):
        result += (ys[j] + ys[j + 1]) / 2 * h
    return result


left_results = []
right_results = []
ns = [2 ** i for i in range(15, 27)]
for n in ns:
    print(np.log2(n), end=' ')
    left_results.append(np.abs(integration_from_left(EPSILON, n) - EXACT_SOLUTION))
    right_results.append(np.abs(integration_from_right(EPSILON, n) - EXACT_SOLUTION))
plt.plot(ns, left_results, label="left")
plt.plot(ns, right_results, label="right")
plt.legend()
plt.grid()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("ns")
plt.ylabel("error")
plt.show()

