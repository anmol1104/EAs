from platypus import NSGAII, NSGAIII, DTLZ2, Hypervolume, experiment, calculate, display, Problem, Real
import numpy as np


if __name__ == "__main__":
    def schaffer(x):
        f1 = x[0] ** 2
        f2 = (x[0] - 2) ** 2
        return np.array([f1, f2])


    algorithms = [NSGAII]
    problem = Problem(1, 2)
    problem.types[:] = Real(-10, 10)
    problem.function = schaffer
    problems = [problem]
    results = experiment(algorithms, problems, nfe=10000, seeds=1)
    print((results['NSGAII']['Problem'][0][0]))


    # calculate the hypervolume indicator
    hyp = Hypervolume(minimum=[0, 0], maximum=[100, 100])
    hyp_result = calculate(results, hyp)
    display(hyp_result, ndigits=3)
