import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')#Only for Mac
import matplotlib.pyplot as plt

'\' m-TSP is solved by making m copies of depot, and then solving as 1-TSP. This big tour can then be broken at each ' \
    'depot to get the m routes.\''

m = int(input('What is the fleet size?: '))
file = np.loadtxt('tsp-48.txt')
nodes = [file[i] for i in range(len(file))]
cities = [i for i in range(len(nodes))]
depot = 4
depot_coord = nodes[depot]
for i in range(m-1):
    nodes.append(depot_coord)
    cities.append(depot)
nodes = np.array(nodes)
N = len(cities)


def cost(z):
    z = np.append(z, z[0])  # Appending fist city at the last to complete the tour
    d = np.diff(nodes[z], axis=0)
    d = np.sqrt((d ** 2).sum(axis=1))
    d[d == 0] = 100000      # A penalty to avoid copies of depot being adjacent to each other in the route
    return (d / np.sqrt(10)).sum()


# TODO: is adding penalty the best way to avoid depots adjacent to each other, or is there a better way?
# Current method works fine

def tweak(x):
    i, j = np.random.randint(0, N, 2)
    while j == i:
        j = np.random.randint(0, N)
    if j < i:
        i, j = j, i
    x[(i + 1):j] = x[(i + 1):j][::-1]   # Intermediate path is reversed
    return x


n = 50000    # number of function evaluations


def TSP():
    # Initialization
    s = np.copy(cities)
    np.random.shuffle(s)
    best = s
    for i in range(n):
        #   Mutation
        r = tweak(np.copy(s))
        #   Evaluation
        if cost(r) < cost(best):
            s = r
            best = r
    return best, cost(best)


seeds = 10


def run(prnt = False):
    route_seed = []
    cost_seed = []
    for i in range(seeds):
        np.random.seed(i + 1)
        z = TSP()
        route_seed.append(z[0])
        cost_seed.append(z[1])
        if prnt:
            print('Seed: ', i + 1)
            print('The routes are: ', z[0])
            print('The cost is: ', z[1], '\n')

    z = list(route_seed[cost_seed.index(min(cost_seed))]), min(cost_seed)
    route = z[0]    # The big optimal route with copies of depot
    S = []          # Set of routes
    ix = []         # indices for depot location
    prev_ix = 0

    for i in range(m):
        depot_ix = route[prev_ix:].index(depot)
        depot_ix += prev_ix
        prev_ix = depot_ix + 1
        ix.append(depot_ix)

    for i in range(m-1):
        S.append(route[ix[i]:(ix[i + 1] + 1)])

    j = i + 1
    if m == 1:
        j = i
    S.append(route[ix[j]:] + route[:(ix[0] + 1)])

    for i in range(m):
        d = np.diff(nodes[S[i]], axis=0)
        d = np.sqrt((d ** 2).sum(axis=1))
        d[d == 0] = 100000  # A penalty to avoid copies of depot being adjacent to each other in the route
        lc = (d / np.sqrt(10)).sum()  # Leg cost
        print('Route #', i + 1, ": ", S[i])
        print('Route cost #', i + 1, ": ", lc)
    print('Total cost: ', z[1])

    x = [nodes[z[0][i]][0] for i in range(len(z[0]))]
    x.append(x[0])
    y = [nodes[z[0][i]][1] for i in range(len(z[0]))]
    y.append(y[0])
    plt.plot(x, y)
    plt.show()


run()

