"""Gradient descent"""


import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')#Only for Mac
import matplotlib.pyplot as plt


def f(x,y):
    return 0.26 * (np.square(x) + np.square(y)) - 0.48 * (x * y)

def fx_prime(x,y):
    return 2 * 0.26 * x - 0.48 * y

def fy_prime(x,y):
    return 2 * 0.26 * y - 0.48 * x

def grd_descent(x_initial,y_inital,alpha,n): # (intial value of x, initial value of y, alpha, number of iterations)
    x = x_initial
    y = y_inital
    xy = [] # Stores x and y values at each iteration
    Z = [] # Stores objective function value at each iteration

    xy.append ([x, y])
    Z.append (f (x, y))
    for iter in range(n):
        x -= alpha * fx_prime(x,y)
        y -= alpha * fy_prime(x,y)
        xy.append([x,y])
        Z.append(f(x,y))
    return xy, Z

output = grd_descent(4, 2, 0.1, 500)
xy = output[0]
Z = output[1]
iterations = len(Z)
print('The minimum of the Matyas funciton is ',min(Z),' attained at ', xy[Z.index(min(Z))])
plt.plot([i + 1 for i in range(iterations)], Z)
plt.xlabel('Iterations')
plt.ylabel('Objective function value')
plt.title('Plot of convergence')
plt.show()

alpha = [0.001,0.01,0.1,1]
for i in range(len(alpha)):
    output = grd_descent (4, 2, alpha[i], 500)
    xy = output[0]
    Z = output[1]
    iterations = len(Z)
    print ('The minimum of the Matyas funciton is ', min (Z), ' attained at ', xy[Z.index (min (Z))])
    plt.plot([i + 1 for i in range(iterations)], Z)
    plt.xlabel('Iterations')
    plt.ylabel('Objective function value')
    plt.title('Plot of convergence for different alphas')
plt.legend(alpha)
plt.show()
