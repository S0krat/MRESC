import numpy as np
import matplotlib.pyplot as plt


def func(x: float) -> float:
    return np.exp(-x) / x


def integration_from_left(eps: float, n: int) -> float:
    h = (10 - eps) / (n - 1)
    xs = [eps + h * i for i in range(n)]
    ys = [func(x) for x in xs]
    result = 0.
    for j in range(n - 1):
        result = result + (ys[j] + ys[j + 1]) / 2 * h
    return result


def integration_from_right(eps: float, n: int) -> float:
    h = (10 - eps) / (n - 1)
    xs = [eps + h * i for i in range(n)]
    ys = [func(x) for x in xs]
    result = 0.
    for j in range(n - 2, -1, -1):
        result = result + (ys[j] + ys[j + 1]) / 2 * h
    return result


left_results = []
right_results = []
epsilons = [2 ** (-i) for i in range(30)]
for epsilon in epsilons:
    left_results.append(integration_from_left(epsilon, 100))
    right_results.append(integration_from_right(epsilon, 100))
plt.plot(epsilons, left_results, label="left")
plt.plot(epsilons, right_results, label="right")
plt.legend()
plt.grid()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("epsilon")
plt.ylabel("result")
plt.show()

