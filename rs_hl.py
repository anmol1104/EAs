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

def random_search(n):
    np.random.seed(1)
    best = np.random.uniform(-32.768,32.68,2)
    ft = []
    ft.append(ackley(best))
    for i in range(n):
        x = np.random.uniform(-32.768,32.68,2)
        if ackley(x) <= ackley(best):
            best = x
        ft.append(ackley(best))
    plt.semilogx([j for j in range(n + 1)], ft)
    plt.show()
    return best, ackley(best)


def hill_climbing(n,sd):
    np.random.seed(1)
    best = np.random.uniform(-32.768, 32.68, 2)
    ft = []
    ft.append(ackley(best))
    for i in range(n):
        x = best + np.random.normal(0, sd, 2)
        if ackley(x) <= ackley(best):
            best = x
        ft.append(ackley(best))
    plt.semilogx([j for j in range(n + 1)], ft)
    plt.show()
    return best, ackley(best)


print(random_search(10000))
print(hill_climbing(10000,1))