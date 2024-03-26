ALPHA_C = 0.9
P_C = 6
X_0 = 1


def iter_pr(x_k):
    return ALPHA_C * x_k + 1 / (x_k ** P_C)


def M_0k_plus(diff):
    pass


x = iter_pr(X_0)
for _ in range(100):
    x = iter_pr(x)

print(x)
