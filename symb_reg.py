import numpy as np
from inspect import signature
from copy import *
import matplotlib as mpl
mpl.use('TkAgg')#Only for Mac
import matplotlib.pyplot as plt


class Node:
    def __init__(self, key):
        self.value = key
        self.parent = None
        self.children = []
        self.numchild = 0

    def add_child(self, child):     # child must be a Node
        if child == self:
            child = Node(child.value)
        self.children.append(child)
        child.parent = self
        self.numchild += 1

    def remove_child(self, child):
        if child not in self.children:
            error = 'Parent-Child mismatch'
            return error
        child.parent = None
        self.children.remove(child)
        self.numchild -= 1

    def replace(self, new_key):
        self.value = new_key


def traverse(n, preorder=True, node=True, countleaf=False, typecheck=False, depthcheck=False):
    if depthcheck:
        countleaf = False   # depthcheck overrides countleaf traverse
        typecheck = False   # depthcheck overrides typecheck

        if n.numchild == 0:
            return 1

        child_depth = []
        for i in range(n.numchild):
            child_depth.append(traverse(n.children[i], depthcheck=True))
        child_depth = np.max(child_depth)
        depth = 1 + child_depth

        return depth

    elif typecheck:
        countleaf = False   # typecheck overrides countleaf traverse
        typeof = []
        if preorder:
            typeof.append(type(n))
        for i in range(n.numchild):
            subtypeof = traverse(n.children[i], preorder, node, False, True)
            for j in range(len(subtypeof)):
                typeof.append(subtypeof[j])
        if not preorder:
            typeof.append(n)
        return typeof

    elif countleaf:
        leafnode = 0
        for i in range(n.numchild):
            leafnode += traverse(n.children[i], preorder, node, True)
        if n.numchild == 0:
            return 1
        return leafnode

    else:
        tree = []
        if node:
            val = n
        else:
            val = n.value

        if preorder:
            tree.append(val)

        for i in range(n.numchild):
            subtree = traverse(n.children[i], preorder, node, False)
            for j in range(len(subtree)):
                tree.append(subtree[j])
        if not preorder:
            tree.append(val)
        return tree


def f(fn, x):   # Function as node and input to this function x as array
    x = copy(x)
    inputs = []
    for i in range(fn.numchild):
        sub_fn = fn.children[i]
        inputs.append(f(sub_fn, x))
    if fn.numchild == 0:
        if fn.value in identity_set:
            ix = identity_set.index(fn.value)
            t = x[ix]
            inputs.append(t)
        elif fn.value.__name__ == 'constant':
            inputs.append(0)
    return fn.value(*inputs)


identity_x = lambda x: x  # TODO: Should create as many identity functions as the dimension of the problem*
identity_x.__name__ = 'identity_x'
identity_set = [identity_x]   # TODO: All the identity function need to be added here
constant = lambda x: 1
constant.__name__ = 'constant'
add = lambda x,y: np.add(x,y)
add.__name__ = 'add'
subtract = lambda x,y: np.subtract(x,y)
subtract.__name__ = 'subtract'
multiply = lambda x,y: np.multiply(x,y)
multiply.__name__ = 'multiply'
divide = lambda x,y: np.divide(x,y)
divide.__name__ = 'divide'
sin = lambda x: np.sin(x)
sin.__name__ = 'sin'
cos = lambda x: np.cos(x)
cos.__name__ = 'cos'
tan = lambda x: np.tan(x)
tan.__name__ = 'tan'
exp = lambda x: np.exp(x)
exp.__name__ = 'exp'

function_set = [constant, identity_x, add, subtract, multiply, divide, sin, cos, tan, exp]

N = 25        # Number of individuals in a population
gens = 25     # Number of generation the EA is to be run
k = 2           # Number of competitors in the tournament
CR = 0.6        # Crossover probability
mp = 0.125      # Mutation probability


data = []
mystery_fn = lambda x: 2*x**2 + x + 2
for i in range(1000):
    data.append([i/100, mystery_fn(i/100)])

data_x = [[data[i][0]] for i in range(len(data))]
data_y = [data[i][1] for i in range(len(data))]
data_grad = [(data_y[i + 1] - data_y[i])/(data_x[i + 1][0] - data_x[i][0]) for i in range(len(data)-1)]


def EA(m, method):
    if method == 'output':
        boolean = False
    elif method == 'gradient':
        boolean = True

    # Initialization
    P = [rand_fn(np.random.randint(1,m)) for _ in range(N)]
    Best = None

    for gen in range(gens):
        # Evaluation
        for i in range(N):
            model = copy(P[i])
            if Best is None or obj_fn(model, boolean) < obj_fn(Best, boolean):
                Best = model

        # Selection
        Q = []
        for i in range(N):
            l = list(range(N))
            np.random.shuffle(l)
            ix = l[:k]
            Q.append(tournament(P, ix, boolean))

        # Crossover and Mutation
        i = 0
        while i in range(N-1):
            x1 = Q[i]
            x2 = Q[i + 1]
            # Crossover
            if np.random.rand() < CR:
                children = crossover(x1, x2)
                P[i] = children[0]
                P[i + 1] = children[1]
            else:
                P[i] = copy(x1)
                P[i + 1] = copy(x2)

            # Mutation
            P[i] = mutate(P[i])
            P[i + 1] = mutate(P[i + 1])

            i += 2

    return Best


def rand_fn(m):
    if m == 1:
        choice_set = ['identity', 'constant']
        choice = choice_set[np.random.randint(0, len(choice_set))]
        if choice == 'identity':
            ix = np.random.randint(0, len(identity_set))
            P = Node(identity_set[ix])
        else:
            P = Node(constant)
    else:
        P = Node(function_set[np.random.randint(0, len(function_set))])

    if (P.value in identity_set) or (P.value == constant):
        return P

    l = args(P.value)
    for j in range(l):
        node = rand_fn(m-1)
        P.add_child(node)
    return P


def args(func):
    sig = signature(func)
    return len(sig.parameters)


arg_fset = [args(function_set[i]) for i in range(len(function_set))]


def obj_fn(x, grad=False):
    est_y = est(x)
    if grad:
        y = data_grad
        est_grad = [(est_y[i + 1] - est_y[i])/(data_x[i + 1][0] - data_x[i][0]) for i in range(len(data)-1)]
        z = [(est_grad[i] - y[i]) ** 2 for i in range(len(data)-1)]
    else:
        y = data_y
        z = [(est_y[i] - y[i])**2 for i in range(len(data))]
    rmse = np.sqrt(np.mean(z))
    return rmse


def est(fn):
    z = []
    data_x = [[data[i][0]] for i in range(len(data))]
    x = data_x
    for i in range(len(x)):
        z.append(f(fn, x[i]))
    return z


def tournament(P, ix, boolean):
    candidates = [P[ix[i]] for i in range(k)]
    z = [obj_fn(candidates[i], boolean) for i in range(k)]
    index = z.index(min(z))
    winner = candidates[index]
    return winner


def mutate(x):
    z = traverse(x)
    for i in range(len(z)):
        if (z[i].value not in identity_set) and (z[i].value.__name__ != 'constant'):
            if np.random.rand() < mp:
                if args(z[i].value) in arg_fset:
                    fn = function_set[np.random.randint(0, len(function_set))]
                    while (fn in identity_set) or (fn.__name__ in ['constant', z[i].value.__name__]) or (args(z[i].value) != args(fn)):
                        fn = function_set[np.random.randint(0, len(function_set))]
                    z[i].replace(fn)

        elif z[i].value in identity_set:
            if len(identity_set) > 1:
                if np.random.rand() < mp:
                    z_ix = identity_set.index(z[i].value)
                    fn_ix = np.random.randint(0, len(identity_set))
                    while fn_ix == z_ix:
                        fn_ix = np.random.randint(0, len(identity_set))
                    fn = identity_set[fn_ix]
                    z[i].replace(fn)

        else:
            if np.random.rand() < mp:
                c = np.random.uniform(0, 180)
                c = c*np.pi/180
                c = tan(c)
                const_fn = lambda x: c
                const_fn.__name__ = 'constant'
                z[i].replace(const_fn)
    return x


def crossover(m, n):
    m = deepcopy(m)
    n = deepcopy(n)
    x = traverse(m)
    y = traverse(n)
    ix, iy = np.random.randint(0, len(x)), np.random.randint(0, len(y))
    subtree_x, subtree_y = x[ix], y[iy]
    parent_x, parent_y = subtree_x.parent, subtree_y.parent
    if parent_x is None:
        m = subtree_y
    else:
        parent_x.remove_child(subtree_x)
        parent_x.add_child(subtree_y)
    if parent_y is None:
        n = subtree_x
    else:
        parent_y.remove_child(subtree_y)
        parent_y.add_child(subtree_x)
    return m, n


def extract(m):
    np.random.seed(4)
    method = 'output'
    model = EA(m, method)
    rmse = obj_fn(model)
    z = traverse(model, node=False)
    z = [z[i].__name__ for i in range(len(z))]
    return model, z, rmse


seeds = 10
depth = 5


def run():
    for i in range(seeds):
        print('Seed: ', i + 1)
        model = []
        func = []
        score = []
        for j in range(1, depth):
            np.random.seed(i + 1)
            z = extract(j + 1)
            model.append(z[0])
            func.append(z[1])
            score.append(z[2])
        ix = score.index(min(score))
        print('The best fit is: ', func[ix])
        t = traverse(model[ix])
        for i in range(len(t)):
            if t[i].value.__name__ == 'constant':
                print('Constant value', t[i].value(0))

        print('RMSE: ', score[ix])
        print('Max tree depth: ', ix + 2, '\n')

        plt.plot(data_x, est(model[ix]))
    plt.plot(data_x, data_y, 'black')
    plt.show()


run()


# TODO: Test of the two methods - gradient and output are actually different



