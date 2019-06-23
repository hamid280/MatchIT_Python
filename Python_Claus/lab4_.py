from math import *

from matplotlib.pyplot import *
from scipy import *


# task 1

def f(alpha, r):
    return r * exp(1j * alpha)


x_values = []
y_values = []

for r in linspace(0.1, 1.0, 5):
    for i in linspace(0, radians(360), 100):
        z = f(i, r)
        x_values.append(z.real)
        y_values.append(z.imag)

plot(x_values, y_values, "ro")
show()


# task 2

def f(x):
    return x ** 3


# function that calculates the derivative of f
def der(f, x, h):
    return (f(x + h) - f(x)) / h


def fp(x):
    return der(f, x, 1.e-9)


def newton(f, fp, x, Tol):
    conv = False
    x_initial = x

    for i in range(0, 400):
        x_final = x_initial - f(x_initial) / fp(x_initial)

        if abs(x_final - x_initial) < Tol:
            conv = True
            return x_initial, conv

        x_initial = x_final

    else:
        print("error because the sequence diverges after the given iterations")


result, conv = newton(f, fp, 1.0, 1e-6)

print(f"The value of xn+1 = {result} and does the sequence converge ? {conv}")
