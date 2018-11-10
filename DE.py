"""Differential Evolution"""


import numpy as np
import matplotlib.pyplot as plt

d = 2   # dimension
lb = -32.768
ub = 32.768


def ackley(x):
    if type(x) not in (list, np.ndarray):
        return None
    x = np.array(x)
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    return -a * np.exp(-b * np.sqrt((1/d) * np.sum(np.square(x)))) - np.exp((1/d) * np.sum(np.cos(c * x))) + a + np.exp(1)


def peaks(x):
    a = 3*(1-x[0])**2*np.exp(-(x[0]**2) - (x[1]+1)**2)
    b = 10*(x[0]/5 - x[0]**3 - x[1]**5)*np.exp(-x[0]**2-x[1]**2)
    c = (1/3)*np.exp(-(x[0]+1)**2 - x[1]**2)
    return a - b - c + 6.551    # add this so objective is always positive


def rosenbrock(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


l = 50      # Population size
gens = 200  # Total number of generations
F = 0.9     # Step size
CR = 0.9    # Crossover probability


def diff_ev(fun):
    P = np.random.uniform(lb, ub, (l,d))
    Best = None
    f = []
    for i in range(gens):
        for j in range(l):
            A = P[j]
            if Best is None or fun(A) < fun(Best):
                Best = A
            B, C = P[np.random.randint(0, l-1, d)]
            D = A + F*(B - C)
            for k in range(d):
                rn = np.random.rand()
                if rn > CR:
                    D[k] = A[k]
            if fun(D) < fun(A):
                P[j] = D
        f.append(fun(Best))
    return Best, fun(Best), f


num_seeds = 10
iters = [k+1 for k in range(gens)]


def run(fun):
    for i in range(num_seeds):
        np.random.seed(i + 1)
        z = diff_ev(fun)
        print('Seed: ', i + 1)
        print('Minimum: ', z[1])
        print('At: ', z[0])
        plt.plot(iters, z[2], 'steelblue')
    plt.show()


run(ackley)
run(peaks)
run(rosenbrock)