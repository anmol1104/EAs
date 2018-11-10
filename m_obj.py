import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from platypus import NSGAII, Problem, Real
import pygmo as pg


def schaffer(x):
    f1 = x[0]**2
    f2 = (x[0] - 2)**2
    return np.array([f1, f2])


def mymulti1(x):
    f1 = x[0]**4 - 10*x[0]**2+x[0]*x[1] + x[1]**4 -(x[0]**2)*(x[1]**2)
    f2 = x[1]**4 - (x[0]**2)*(x[1]**2) + x[0]**4 + x[0]*x[1]
    return np.array([f1, f2])


fun = schaffer
def dominance(A, B):
    if (fun(A) < fun(B)).any():
        if (fun(A) > fun(B)).any():
            return 0
        else:
            return 1
    else:
        return -1


def tournament(x, y):
    if dominance(x, y) == 1:
        return x
    elif dominance(y, x) == 1:
        return y
    else:
        return x


ub = 10
lb = -10
nfe = 2500


def MOEA1():
    np.random.seed(1)
    x = np.random.uniform(lb, ub, d)
    archive = [x]
    for i in range(nfe):
        x = np.random.uniform(lb, ub, d)
        z = np.array([dominance(x, archive[j]) for j in range(len(archive))])
        if (z == 0).all():
            archive.append(x)
        if (z == 1).any():
            ix = np.array([j for j, val in enumerate(z) if val == 1])
            archive = np.array(archive)
            archive[ix] = x
            archive = list(archive)
    return archive


N = 10
num_gens = 100
CR = 1
mp = 1
s = 0.5
if fun == schaffer:
    d = 1
elif fun == mymulti1:
    d = 2


def MOEA2():
    np.random.seed(1)
    P = np.random.uniform(lb, ub, (N, d))
    archive = [P[0]]
    for gen in range(num_gens):
        for i in range(N):
            z = np.array([dominance(P[i], archive[j]) for j in range(len(archive))])
            if (z == 0).all():
                archive.append(copy(P[i]))
            if (z == 1).any():
                ix = [j for j, val in enumerate(z) if val == 1]
                archive = np.array(archive)
                archive[ix] = copy(P[i])
                archive = list(archive)

        Q = []
        for i in range(N):
            x1, x2 = np.random.randint(0, N, 2)
            while x2 == x1:
                x2 = np.random.randint(0, N)
            winner = copy(tournament(P[x1], P[x2]))
            Q.append(winner)

        for i in range(N-1):
            c1 = Q[i]
            c2 = Q[i + 1]
            if np.random.rand() < CR:
                ix = np.random.randint(0, d)
                c1[ix:d] = Q[i + 1][ix:d]
                c2[ix:d] = Q[i][ix:d]

            if np.random.rand() < mp:
                c1 = np.clip(c1 + np.random.normal(0, s, d), lb, ub)
            if np.random.rand() < mp:
                c2 = np.clip(c2 + np.random.normal(0, s, d), lb, ub)
            P[i] = c1
            P[i + 1] = c2

    return archive


def main():
    z = MOEA1()
    f = [fun(z[i]) for i in range(len(z))]
    f1 = [f[i][0] for i in range(len(f))]
    f2 = [f[i][1] for i in range(len(f))]
    plt.plot(f1, f2, marker='.', linewidth=0)
    plt.title('Single individual population: No crossover')
    plt.show()
    hv = pg.hypervolume(f)
    print(hv.compute([10, 10]))

    z = MOEA2()
    f = [fun(z[i]) for i in range(len(z))]
    f1 = [f[i][0] for i in range(len(f))]
    f2 = [f[i][1] for i in range(len(f))]
    plt.plot(f1, f2, marker='.', linewidth=0)
    plt.title('Multiple individuals population: Crossover and mutatation')
    plt.show()
    hv = pg.hypervolume(f)
    print(hv.compute([10, 10]))

    problem = Problem(1, 2)
    problem.types[:] = Real(-10, 10)
    problem.function = schaffer
    algorithm = NSGAII(problem)
    algorithm.run(10000)
    f = []
    for solution in algorithm.result:
        z = list(np.copy(solution.objectives))
        f.append(z)

    f1 = [f[i][0] for i in range(len(f))]
    f2 = [f[i][1] for i in range(len(f))]
    plt.plot(f1, f2, marker='.', linewidth=0)
    plt.title('Platypus solution')
    plt.show()
    hv = pg.hypervolume(f)
    print(hv.compute([10, 10]))



if __name__ == '__main__':
    main()