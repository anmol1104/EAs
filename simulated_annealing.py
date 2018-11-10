import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')#Only for Mac
import matplotlib.pyplot as plt

def ackley(x):
    if type(x) not in (list, np.ndarray):
        return None
    x = np.array(x)
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    return -a * np.exp(-b * np.sqrt((1/d) * np.sum(np.square(x)))) - np.exp((1/d) * np.sum(np.cos(c * x))) + a + np.exp(1)


lb = -32.768
ub = 32.768
d = 2 # dimension
n = 10000 # number of function evaluations
def sa(sd, t, alpha):
    np.random.seed(2)
    xt = []
    ft = []

    s = np.random.uniform(lb, ub, d)
    f = ackley(s)
    xt.append(s)
    ft.append(f)
    best = s
    for i in range(n):
        r = np.clip(s + np.random.normal(0, sd, d), lb, ub)
        rn = np.random.rand()
        if ackley(r) < ackley(s):
            s = r
            best = s
        elif rn < p(t, r, s):
            s = r
        t = t*alpha
        f = ackley(s)
        xt.append(s)
        ft.append(f)
    plt.semilogx([j for j in range(n + 1)], ft)
    plt.show()
    return ackley(best), best


def p(t, r, s):
    return np.exp((ackley(s) - ackley(r))/t)


print(sa(1, 100, 0.95))
