import numpy as np

def ackley(x):
    if type(x) not in (list, np.ndarray):
        return None
    x = np.array(x)
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    return -a * np.exp(-b * np.sqrt((1/d) * np.sum(np.square(x)))) - np.exp((1/d) * np.sum(np.cos(c * x))) + a + np.exp(1)


n = 100000 # Iterations for hill climbing
d = 2 # Dimension of ackley function
def hill_climbing(sd):
    best = np.random.uniform(-32.768, 32.768, d)
    for i in range(n):
        x = best + np.random.normal(0, sd, d)
        if ackley(x) <= ackley(best):
            best = x
    return best, ackley(best)


m = 100 # Iterations for gradient descent
def gradient_descent(x, alpha):
    #gradient = ackley_prime(x)
    #while gradient.all() != 0:
    for i in range(m):
        gradient = ackley_prime(x)
        x -= alpha * gradient
    return x, ackley(x)


def ackley_prime(x):
    d = len(x)
    delta_x = 0.001
    gradient = []
    for i in range(d):
        delta_x = np.array([0 if j != i else delta_x for j in range(d)])
        a = ackley(x - delta_x)
        b = ackley(x + delta_x)
        delta_x = delta_x[i]
        gradient.append((b - a)/(2 * delta_x))
    return np.array(gradient)


def hybrid(sd, alpha):
    best = hill_climbing(sd)[0]
    print('Phase 1 (post hill-climbing): ', best, ackley(best))
    best = gradient_descent(best, alpha)[0]
    print('Phase 2 (final results): ', best, ackley(best))


def run(seeds):
    for i in range(seeds):
        np.random.seed(i + 1)
        print('Seed: ', i + 1)
        hybrid(0.5, 0.0001)

run(10)