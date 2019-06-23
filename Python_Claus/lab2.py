from scipy import *
from matplotlib.pyplot import *

# task1


x = 0.5
a = 8

for i in range(200):
    if abs(sin(x) - a * x + 30 - x) < 1.e-8:
        break
    if i == 199:
        if abs(sin(x) - a * x + 30 - x) > 1.e-8:
            print("The condition was not met within 200 iteration steps")

    x = sin(x) - a * x + 30

print("The result after {num} iterations is {res}".format(num=i, res=x))

# task2


x_values = []
y_values = []
a = 0.5

for x in range(5, 31):
    y = sin(x) - a * x + 30

    x_values.append(x)
    y_values.append(y)

    print(x)
    print("y ", y)

plot(x_values, y_values, "o")
# show()


# task3

x_values = []
for i in range(1, 5000000):
    x = sin(i) ** 2 / i
    x_values.append(x)
    print(x)

    if x < 1.e-9:
        break

print(f"The list is  {len(x_values)} elements long")

# task4


print("#################################task 4 solution ####################3 ")
for a in [-0.5, 0.5, -0.25, 0.25]:
    x = 1.0  # given in the text
    for i in range(30):  # we loop till 30

        x1 = 0.2 * x - a * (x ** 2 - 5)
        # print(i, x)
        if abs(x1 - x) < 1.e-9:
            print(f" The sequence converges when a = {a}, n = {i}, and x = {x}")
            break
        x = x1
    else:
        print("No convergence detected")

# task 5

x = 1.0  # given in the text
L = [x]

for i in range(30):
    x1 = 0.2 * x - 0.5 * (x ** 2 - 5)  # a = 0.5
    if abs(x1 - x) < 1.e-9:
        break
    x = x1
    L.append(x)

# now we construct two lists to separate negative and positive values from L

L_negative = [e for e in L if e < 0]  # construct list from elements which are negative
L_positive = [e for e in L if e > 0]  # construct list from elements which are positive

print("Positive elements of the list are : ", L_positive)
print("Negative elements of the list are : ", L_negative)


# task 6

def check_convergence(a):
    x = 1.0

    for i in range(30):
        x1 = 0.2 * x - a * (x ** 2 - 5)  # a = 0.5
        if abs(x1 - x) < 1.e-9:
            return True
        x = x1

    else:
        return False


print(check_convergence(0.5))
print(check_convergence(-0.5))
print(check_convergence(0.25))
print(check_convergence(-0.25))

# ask the teacher
# When we execute our function with a = 1.0. It gives the error (OverflowError: (34, 'Numerical result out of range')

print(check_convergence(1.0))


# task 7

def check_convergence(a, x0):
    for i in range(30):
        x1 = 0.2 * x0 - a * (x0 ** 2 - 5)
        if abs(x1 - x0) < 1.e-9:
            return True
        x0 = x1

    else:
        return False


print(check_convergence(0.5, 1.0))
print(check_convergence(-0.5, 1.0))
print(check_convergence(0.25, 1.0))
print(check_convergence(-0.25, 1.0))


# task 8

def check_convergence(a, x0):
    pos = []  # The list that will contain positive elements of the sequence
    neg = []  # The list that will contain negative elements of the sequence
    for i in range(30):
        x1 = 0.2 * x0 - a * (x0 ** 2 - 5)
        if abs(x1 - x0) < 1.e-9:
            result = True
            break

        if x1 > 0:
            pos.append(x1)
        else:
            neg.append(x1)

        x0 = x1

    else:
        result = False

    print("list of positive elements of the sequence : \n", pos)
    print("list of negative elements of the sequence : \n", neg)
    print("Is sequence divergent ? ")
    return result


print(check_convergence(0.5, 1.0))
print(check_convergence(-0.5, 1.0))
print(check_convergence(0.25, 1.0))
print(check_convergence(-0.25, 1.0))
