"""Genetic Algorithms"""


import numpy as np
import random
import copy


def onemax(x):
    return np.sum(x)


w = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]
v = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
c = 165


def knapsack(x):
    if np.sum(x*w) > c:
        return 0
    else:
        return np.sum(x*v)


k = 2   # Tournament candidates
N = 10  # Population size
CR = 0.6    # Crossover probability
mp = 0.125  # Mutation probability
gens = 80  # Number of generations to run the GA


def GA(fun):
    if fun == knapsack:
        d = 10
    else:
        d = 8

    #Initialization
    P = np.array([[random.randint(0, 1) for j in range(d)] for i in range(N)])
    Best = None

    for gen in range(gens):
        for i in range(N):  # Evaluations
            if Best is None or fun(Best) < fun(P[i]):
                Best = copy.copy(P[i])

        Q = []
        for i in range(N):  # Tournament selection
            ix = random.sample(range(0, N), k)
            f = [fun(P[ix[j]]) for j in range(k)]
            winner = copy.copy(P[ix[np.argmax(f)]])
            Q.append(winner)

        for i in range(N - 1):  # Crossover and mutation
            x1 = Q[i]
            x2 = Q[i + 1]
            if random.random() < CR:   # Crossover
                m = random.randint(0, d - 1)
                P[i] = np.array(list(x1[:m]) + list(x2[m:]))
                P[i + 1] = np.array(list(x1[m:]) + list(x2[:m]))
            else:
                P[i] = copy.copy(Q[i])

            for j in range(d):  # Mutation
                if random.random() < mp:
                    P[i][j] = 1 - P[i][j]
                if random.random() < mp:
                    P[i + 1][j] = 1 - P[i + 1][j]

    return Best, fun(Best)


print(GA(onemax))
print(GA(knapsack))