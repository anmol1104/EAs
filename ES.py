import numpy as np


def ackley(x):
    if type(x) not in (list, np.ndarray):
        return None
    x = np.array(x)
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    return -a * np.exp(-b * np.sqrt((1 / d) * np.sum(np.square(x)))) - np.exp(
        (1 / d) * np.sum(np.cos(c * x))) + a + np.exp(1)


def peaks(x):
    if x is None:
        return None
    a = 3*(1-x[0])**2*np.exp(-(x[0]**2) - (x[1]+1)**2)
    b = 10*(x[0]/5 - x[0]**3 - x[1]**5)*np.exp(-x[0]**2-x[1]**2)
    c = (1/3)*np.exp(-(x[0]+1)**2 - x[1]**2)
    return a - b - c + 6.5511 # add this so objective is always positive


s = 0.25     # standard deviation
l = 50      # Lambda
m = 10      # Mu
P = []      # Population
gens = 500


def es(fun):
    if fun == peaks:
        d = 2
        lb = -3
        ub = 3

    else:
        d = 2
        lb = -32.768
        ub = 32.768

    P = np.random.uniform(lb, ub, (l,d))
    Best = None

    for i in range(gens):
        f = []
        for j in range(l):
            f.append(fun(P[j]))

        ix = np.argsort(f)[:m]
        Q = P[ix]

        fitness_best = fun(Best)
        if fitness_best is None or f[ix[0]] < fitness_best:
            Best = Q[0]

        child = 0
        for j in range(m):
            for k in range(int(l / m)):
                P[child] = np.clip(Q[j] + np.random.normal(0, s, d), lb, ub)
                child += 1
    return Best, fitness_best

print(es(ackley))