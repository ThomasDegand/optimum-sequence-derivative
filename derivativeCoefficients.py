import numpy as np
import matplotlib.pyplot as plt

M = 14
CH = np.zeros((M,M))
ONE = np.ones((M,1))
mu = 0.271828182846

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
T = np.linspace(-2.5,2.5,1000)
SH = fbis(T)
print("p correlation: ", np.corrcoef(T,SH)[0][1])
plt.plot(T,SH)
plt.axis((-2.5,2.5,-2.5,2.5))
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

T2 = np.linspace(-10,10,8000)
plt.plot(T2, np.cos(T2)) #sin derivative
plt.plot(T2, df(np.sin,T2)) #estimated derivative
plt.show()
print("cos correlation: ", np.corrcoef(np.cos(T2),df(np.sin,T2))[0][1])
