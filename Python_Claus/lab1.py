from scipy import *
from matplotlib.pyplot import *
import sys
from math import atan
from math import sqrt


# task 2
def myfun(x):
    print(int(x ** 2 + 0.25 * x - 5))


myfun(2.3)

# task3
L = [1, 2]
L3 = 3 * L

L3[0]

L3[-1]

# L3[10]          # list index out of range


# task4
L4 = [k ** 2 for k in L3]

print("""
      
      L4 :  ###########################################################################
      
      """)
print(L4)

# task5

L5 = L3 + L4

# task 6

L6 = [x / 99 for x in range(100)]
print("""
      
      L6 :  ###########################################################################
      
      """)

print(L6)

# teachers solution







# task7
s = 0
for i in range(0, 500):
    s = s + i
print(s)

print("""
      
      ss list :  ###########################################################################
      
      """)
ss = [0]
for i in range(1, 500):
    ss.append(ss[i - 1] + i)
print(ss)

# last element in the list s and ss were the same. it was 124750.
# value of i after execution of for was 499


# task8

# could not figure out how to use counter in this exercise
xplot = []
for number in L6:
    xplot.append(number)

print("""
      
      xplot :  ###########################################################################
      
      """)

print(xplot)

# task9

# to use arctangent

print("""
      
      Yplot :  ###########################################################################
      
      """)

yplot = [atan(x) for x in xplot]
print(yplot)

# task10
print("""
      
      xplot, yplot graph :  ###########################################################################
      
      """)

xlabel("xplot")
ylabel("yplot")
plot(xplot, yplot)
show()

# task 11

mysum = 0
for i in range(1, 201):
    mysum += 1 / sqrt(i)

print("""
      
      sum of numbers :  ###########################################################################
      
      """)

print(mysum)
