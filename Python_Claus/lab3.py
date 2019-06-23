from scipy import *
from matplotlib.pyplot import *

# task1

L = [0, 1, 2, 1, 0, -1, -2, -1, 0]
K = [1, 2]

print(L[0])
print(L[-1])
print(L[:-1])
print(L + L[1:-1] + L)
K[2:2] = [-3]
print("K is", K)
L[2:2] = [-3]
print("L 2:2 is", L)
L[2:3] = [-3]
print("L 2:3 is", L)
L[3:4] = []
print("L is", L)
L[2:5] = [-5]
print("L is", L)


# task 2

def f(x):
    return sin(x)


# does this work?

x = 3.
# print(f())  # TypeError: f() missing 1 required positional argument: 'x'
# and this?
print(f)

y = 2 * pi
print(f(y))


# task 3
def f(m):
    return 0.0


# task 4

L1 = [0, 20, 30, 40]
L2 = [20, 0, 50, 60]
L3 = [30, 50, 0, 70]
L4 = [40, 60, 70, 0]

distance = [L1, L2, L3, L4]
print("Distance = ", distance)

# first solution using for loop
reddistance = []
for i in range(1, len(distance)):
    reddistance.append(distance[i][0:i])

print("reddistance", reddistance)

# second solution using for loop
reddistance2 = []

for i in range(len(distance)):
    mylist = []
    for j in range(len(distance[i])):
        if i > j:
            mylist.append(distance[i][j])
    if mylist:
        reddistance2.append(mylist)

print("reddistance2 = ", reddistance2)

# solution using list slice and list comprehension
reddistance3 = [distance[i][0:i] for i in range(1, len(distance))]
print("reddistance 3 = ", reddistance3)

# another solution

# mylist = []
# for i, l in enumerate(distance[1:]):
#     mylist.append(l[:i+1])
#
# print(mylist)

# task 5

A = {'a', 'b', 'c', 'd'}
B = {'c', 'd', 'e'}

A_B = A - B
B_A = B - A

# symmetric difference of two sets = (a - b) U (b - a)

result1 = A_B.union(B_A)
print(result1)

# using inbuilt symmetric_difference function
result2 = A.symmetric_difference(B)
print(result2)


# task 6


# A.update(B)
# Here, A and B are two sets. The elements of set B are added to the set A.


# The intersection of two or more sets is the set of elements which are common to all sets.
# A.intersection_update(*other_sets)


# task 8

def func(x):
    return 3 * x ** 2 - 5


def func2(x):
    return arctan(x)


def bisec(interval, tolerance):
    a = interval[0]
    b = interval[1]

    if func(a) * func(b) > 0:
        return "No root found."

    else:
        while b - a >= tolerance:
            midpoint = (a + b) / 2.0

            # Check if midpoint is root
            if func(midpoint) == 0.0:
                return midpoint

            # Decide the midpoint to repeat the steps
            if func(midpoint) * func(a) < 0:
                b = midpoint
            else:
                a = midpoint

        interval[0] = a
        interval[1] = b

        return interval, midpoint


f_interval, midpoint = bisec([-1.5, -0.4], 1e-9)

print(f" Final interval is = {f_interval} and the midpoint is = {midpoint}")

# plot the function to justify the results
x = linspace(-1.5, -0.4)
title("polynomial plot")
plot(x, func(x))
show()


# task 9

def bisec(f, interval, tolerance):
    a = interval[0]
    b = interval[1]

    if f(a) * f(b) > 0:
        return "No root found."

    else:
        while b - a >= tolerance:
            midpoint = (a + b) / 2.0

            # Check if midpoint is root
            if f(midpoint) == 0.0:
                return midpoint

            # Decide the midpoint to repeat the steps
            if f(midpoint) * f(a) < 0:
                b = midpoint
            else:
                a = midpoint

        interval[0] = a
        interval[1] = b

        return interval, midpoint


f_interval, midpoint = bisec(func, [-1.5, -0.4], 1e-9)

print(f" Final interval is = {f_interval} and the midpoint is = {midpoint}")

f_interval2, midpoint2 = bisec(arctan, [-0.1, 2], 1e-3)
print(f" Final interval is = {f_interval2} and the midpoint is = {midpoint2}")
