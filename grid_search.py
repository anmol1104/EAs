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


def grid_search(s, d):
    grids = int(np.floor((32.768 - (-32.768))/s))
    grid_pos = [[round(-32.768 + s * j, 3) for j in range(grids + 1)] for i in range(d)]
    xt = []
    ft = []
    for i in range(grids + 1):
        for j in range(grids + 1):
            x = [grid_pos[0][i], grid_pos[1][j]]
            f = ackley(x)
            xt.append(x)
            ft.append(f)
    return min(ft), xt[ft.index(min(ft))]


print(grid_search(0.01, 2))


