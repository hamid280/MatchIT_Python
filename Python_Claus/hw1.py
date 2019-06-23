"""
MatchIT 2019- Homework 1
due date : 31th May 2019

@author:

Hamid SIDDIQI
Aycoul Amotlak
Burcu Boran


"""

from matplotlib.pyplot import *
from numpy import *
from scipy.integrate import quad

list_of_h = []  # we need to store values of h because we need it for plotting the graph (as width of interval)

# task1

print("###################################    task1 solution    ##################")


def ctrapezoidal(f, a, b, n ):
    """
   Return an approximation to the definite integral of f from a to b
   using the trapezium rule with n intervals.
   in other words : Trapezoidal approximation is to divide area of function into n trapeziums and add areas.

     Parameters
   ----------
   f : the provided function
   a : lower limit of the given integral
   b : upper limit of the given integral
   n : number that we are supposed to divide the trapeziums. (Large n value means better approximation)



   Returns
   -------
   I : sum of added areas of the nth trapeziums

    """

    h = float(b - a) / n  # width of the intervals
    list_of_h.append(h)

    I = 0.5 * f(a) + 0.5 * f(b)
    for i in range(1, n):
        I += f(a + i * h)
    I *= h
    return I


# example : for the given values below we calculate trapezoidal and compare the results for given n
a = 0


b = pi
n = 20
f = lambda x: sin(x)

trapezoidal_result = ctrapezoidal(f, a, b, n)
print(f"The result of approximation area for the given function using trapezoidal method is : {trapezoidal_result}")

# task 2
print()
print("###################################    task2 solution    ##################")

given_tolerance = 1e-4
list_of_T = []
list_of_h[:] = []  # reset the list of h

for i in range(1, n + 1):
    T_last = ctrapezoidal(f, a, b, i)
    if len(list_of_T) > 1 and abs(T_last - list_of_T[-1]) < given_tolerance:
        break
    else:
        list_of_T.append(T_last)
print(f"final approximation using trapezoidal method when n = {i} is : {list_of_T[-1]}")

# task 3


print()
print("###################################    task3 solution    ##################")


def plot_loglog():
    """
     plots the error list with accordance to interval width to log scalling

    """

    error_list = []

    # get the exact value from subtracting last point of integral from the starting point
    exact_value, _ = quad(f, a, b)
    for i in range(len(list_of_T)):
        error_list.append(abs((list_of_T[i]) - exact_value))

    error_list = array(error_list)
    interval_width = array(list_of_h)

    xlabel('Log10(h)')
    ylabel('Log10(error)')
    loglog(interval_width, error_list, marker='*')
    grid(True, which="both")
    show()


plot_loglog()

print("The graph is drawn for task 3")

# task4
print()


def sparbanken(K_0, r, n, R):
    """
       Calculates the remaining loan with accordance to the given years and other inputs

         Parameters
       ----------
       K_0: initial value of the loan
       r : the yearly interest
       n : number of years for which the loan is expected to be calculated
       R : The yearly (constant) amortization
    """

    print("###################################    task4 and 5 solution    ##################")
    # check if the given arguments are positive
    if K_0 <= 0 or R < 0 or r < 0:
        print("Please enter only positive numbers!")
        return

    # first remaining loan is considered to be the initial loan
    k = K_0

    # make a list to store the loans
    loans = []

    # the first year when loan is fully paid
    year_no_loan_left = 0

    # calculate the remaining loan for given years n
    for i in range(n + 1):

        # append the loan to the list for every year starting from year 0.
        loans.append(k)

        k_new = k * (r / 100 + 1) - R
        k = k_new

        # when no loan is left
        if k < 0:
            year_no_loan_left = i
            break
    return loans, year_no_loan_left


# example

print("Please enter the amount that you wish to loan (K_0).")
K_0 = float(input())

print("Please enter the yearly interest (r).")
r = float(input())

print("Please enter the yearly amortization rate (R).")
R = float(input())

loans, year_loan_is_paid = sparbanken(K_0, r, n, R)

if year_loan_is_paid > 0:
    print("The first year when all loan is fully paid, is year : ", year_loan_is_paid)
else:
    print(f"Remaining loan after {n} years is : {round(loans[-1])} ")
    print("Please see the graph for this calculation")

print()
print("###################################################################################\n")


# task6
print("Please enter the last year that you want the loans to be fully paid. (n)")
n = int(input())

print()
print("###################################    task6 solution    ##################")


def amortization_rate(K_0, r, n):
    """
       Calculate the amortization rate the given period of years. We use the formula that we had found
       after researching on the internet.
       Formula is: Amortization rate = principal * monthly_interest / 1 - (1 + monthly_interest) ** (-n)

         Parameters
       ----------
       K_0 : The principal or the loaned money
       r : interest rate in percentage
       n : The period of years that we are supposed to calculate the amortization rate.

       Returns
       -------
       R : The amortization rate for the given period of years

    """

    # determining monthly interest rate
    monthly_interest = (1 + r * 0.01) ** (1 / 12) - 1

    # we change year to be in months
    n = n * 12

    # calculate amortization rate using the formula mentioned above
    R = K_0 * monthly_interest / (1 - (1 + monthly_interest) ** (-n))

    # Since we are supposed to find amortization rate for the whole year
    R *= 12

    return round(R)


# example
yearly_amortization = amortization_rate(K_0, r, n)
print(f"Yearly amortization rate for the loan to get paid fully back after {n} years is : {yearly_amortization} kr .")

if yearly_amortization > R:
    print(f"Thus, if you want your loans to be paid fully after {n} years. You have to pay"
          f" {round(yearly_amortization - R)}kr more each year")

# task 7

# 1 sth graph shows the amortization rate vs interest rate. The aim of this graph is to show that how yearly
# amortization rate changes with accordance to the different yearly interest

# task 7

print("See the graph for the changes of amortization rate(R) with accordance to the interest rate (r)")

# 1 sth graph shows the amortization rate vs interest rate. The aim of this graph is to show that how yearly
# amortization rate changes with accordance to the different yearly interest

list_r = [r + 0.5, r + 1., r + 1.5, r + 2., r + 2.5]

# for every value of r we calculate R and append it to the list
list_R = []
for r_new in list_r:
    list_R.append(amortization_rate(K_0, r_new, n))

# change lists to arrays
list_r = array(list_r)
list_R = array(list_R)

# prepare necessary data for the plot of the graph
title(' Interest Rate (r%) vs Amortization Rate (R) ')
xlabel('r(%)')
ylabel('R(kr)')
plot(list_r, list_R, marker='*')
grid()
show()

# Second graph shows the years versus their corresponding loans.

# round each value to integer
loans = [round(loan) for loan in loans]

# change list of loans to array
loans = array(loans)

# get the years with accordance to the number of loans
list_year = array(range(len(loans)))
# prepare to draw the graph
title(' Years (n) vs Remaining loans (K) ')
plot(list_year, loans, marker='*')
xlabel('Years (n)')
ylabel('Loans Remained (kr)')
grid()
show()
