import numpy as np
import matplotlib.pyplot as plt

M = 14
CH = np.zeros((M,M))
ONE = np.ones((M,1))
mu = 0.27

for i in range(1,M+1):
    for j in range(1,M+1):
        CH[i-1][j-1] = np.cosh(mu*i*j)

A = np.dot(np.linalg.inv(CH),ONE)

for i in range(1,M+1):
    A[i-1] /= i

def f(t):
    s = 0
    for i in range(1,M+1):
        s += A[i-1]*np.sinh(i*t)
    return s

B = A/f(1)

def fbis(t):
    s = 0
    for i in range(1,M+1):
        s += B[i-1]*np.sinh(i*t)
    return s


print(B)
b = 2.5
T = np.linspace(-b,b,1000)
SH = fbis(T)
print("p correlation: ", np.corrcoef(T,SH)[0][1])
plt.plot(T,SH)
plt.axis((-b,b,-b,b))
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(color='k', linestyle='--', linewidth=0.5)
plt.show()

def df(g,t):
    s = 0
    for i in range(1,M+1):
        s += B[i-1]*g(t-i)/(-2)
    for i in range(1,M+1):
        s += B[i-1]*g(t+i)/2
    return s

def g0(t):
    return np.sin(t)

def dg0(t):
    return np.cos(t)

T2 = np.linspace(-10,10,8000)
plt.plot(T2, dg0(T2)) #derivative
plt.plot(T2, df(g0,T2)) #estimated derivative
plt.show()
print("derivative correlation: ", np.corrcoef(dg0(T2),df(g0,T2))[0][1])
