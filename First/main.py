import matplotlib.pyplot as plt
import numpy as np
from operator import add
import copy
from mpl_toolkits.mplot3d import Axes3D
import random as rnd
import time


class Profiler(object):
    def __enter__(self):
        self._startTime = time.time()

    def __exit__(self, type, value, traceback):
        print("Elapsed time: {:.3f} sec".format(time.time() - self._startTime))



def multip(l, mul):
    for i in range(len(l)):
        l[i] *= mul
    return l


def Runge(f, tstart, tend, ystart, tau, p, N = 100000):
    Nmax = int(min(N, (tend - tstart) / tau))
    result = [[tstart, ystart]]
    t = tstart
    y = ystart

    for i in range(Nmax):
        k1 = multip(f(y), tau)
        k2 = multip(f(list(map(add, y, multip(k1, tau / (2 * p))))), tau)
        k1mul = multip(k1, 1 - p)
        k2mul = multip(k2, p)
        t += tau
        for j in range(len(y)):
            y[j] += k1mul[j] + k2mul[j]
        result.append([t, copy.copy(y)])

    return result, (tend - tstart) / Nmax


def p(x):
    return 1 + np.sin(np.float64(x) * np.float64(np.pi) / np.float64(2))


def f(y, x, eta, j, n):
    return (y[j+1] - 2*y[j] + y[j-1])/(n * h**2) + eta * y[j] * (1 - y[j] / x[j])


def fVec(y, x, eta, n):
    yNew = [0]
    for i in range(n):
        yNew.append(f(y, x, eta, i+1, n))
    yNew.append(yNew[n-1])
    return copy.copy(yNew)


def RK(fV, y, x, tau, eta, l, n, p):
    yTmp = y[:, l]
    k1 = tau * np.array(fV(yTmp, x, eta, n))
    k2 = tau * np.array(fV(yTmp + tau * k1 / (2*p), x, eta, n))
    yNew = yTmp + (1-p) * k1 + p * k2
    y[:, l+1] = yNew


def solve(h, tau, eta, L):
    N = int(round(1/h))
    h = 1.0/N
    x = np.array(range(N+2)).astype(np.float64)
    for i in range(N+1):
        x[i] = p(h*i)
    x[N+1] = x[N-1]

    y = np.zeros((N+2, L+1))
    y[:, 0] = x

    for i in range(L):
        RK(fVec, y, x, tau, eta, i, N, 0.5)

    return y


def getXY(h, tau, L):
    N = int(round(1/h))
    X = np.ndarray((N+2, L+1))
    Y = X.copy()
    for k in range(N+2):
        for j in range(L+1):
            X[k, j] = k * h
            Y[k, j] = j * tau

    return X, Y


def save_3d(data, X, Y, fname = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, data)
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)


h = 0.05
tau = 0.001
L = 100000
eta = 0
#НАБЛЮДЕНИЕ
#При малом h система "стабилизируется" быстрее, чем при больших
#y = solve(h, tau, eta, L)
#X, T = getXY(h, tau, L)
#print(X.shape)
#print(y.shape)
#save_3d(y, T, X)

n = 20000
sum = 0
with Profiler() as p:
    for i in range(n):
        for j in range(n):
            sum += 1.0 / ((i+1)*(j+1))
    print(sum)

b = np.zeros((n))
for i in range(n):
    b[i] = rnd.uniform(1.0, 1.1)

with Profiler() as p:
    for i in range(n):
        for j in range(n):
            sum += 1.0 / ((b[i])*(b[j]))
    print(sum)

print(2)