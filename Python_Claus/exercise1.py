import dateutil.parser
import numpy as np
import matplotlib.pyplot as plt

# Solution to exercise 1

# We firsts consider an array to be ranged from 10 till 39(inclusive) , after that we try to reshape this
# array as a matrix with 6 rows and 5 columns
from dateutil.parser import parse

M = np.arange(10, 40).reshape(6, 5)
print("Exercise 1 solution")
print(M)

# Solution to exercise 2

# As we have defined the m_kg as a numpy array data type we can do the mathematical operation so simply because its
# factorized. (We can not do this if we use normal inbuilt array function)
m_kg = np.array([50, 100, 200, 1000, 1500])

print()
print("Exercise 2 Solution")
print(m_kg / 0.45359237)

# Solution to exercise 3

random_array = np.random.random((3, 4))
print()
print("Exercise 3 Solution")

print(random_array)

i = 0
for row in random_array:
    print(f"Max and Min values of row at index: {i}  are: {row.max()}, {row.min()} ")
    i += 1

# Solution to exercise 4
print()
print("Exercise 4 Solution")

print("Output, sort along axis=0")
print(np.sort(random_array, axis=0).round(3))

print("Output, sort along axis=1")
print(np.sort(random_array, axis=1).round(3))

# Solution to exercise 5
print()
print("Exercise 5 Solution")

X = np.arange(101)  # array from 0 till 100
my_sum = 0.
for x in X:
    if (x % 3 == 0) or (x % 5 == 0):
        my_sum += x

print(my_sum)

# Solution to exercise 6


my_data = np.array([(1, "Albert Einstein", 1.2),
                    (2, "Muhammad ibn Musa al Khwarizmi", 1.3),
                    (3, "Kurt Godel", 1.4)], dtype=[('order', 'i4'), ('name', 'U40'), ('other', 'f4')])
print()
print("Exercise 6 Solution")
print(my_data.dtype)

# save the data to a file
np.save('saved_data', my_data)

# load the saved data from file
loaded_data = np.load('saved_data.npy')
print(loaded_data)

# solution to question 7

print()
print("Solution to exercise 7")
data = np.genfromtxt('database.csv', delimiter=',', names=True, )


# print(data.dtype)

# part a : every datatype is set as default to f8.
# we are getting nan values because every data type is considered to be f8 and those who cannot be
# converted to f8 is returned as nan


# part b


def convert(dateString):
    return dateutil.parser.parse(dateString, dayfirst=True).year


data = np.genfromtxt('database.csv', delimiter=',', usecols=(0, 5, 8),
                     converters={0: convert}, dtype=[('Date', 'U10'), ('Depth', 'f8'), ('Magnitude', 'f8')],
                     names=True)
print(data)
print(data.dtype)

# part c
# we get only the magnitude column from the file

magnitude = np.genfromtxt('database.csv', delimiter=',', usecols=(8),
                          converters={8: lambda s: float(s or 0)}, dtype=[('Magnitude', 'f8')],
                          names=True)

# data2 = np.genfromtxt('database.csv', delimiter=',')  # To see the default data type
# data = np.genfromtxt('database.csv', delimiter=',', dtype=('unicode'))
# Magnitude = data[1:, 8:9].astype(float)

# print the min value from magnitude column
print("----------------------------------------------")
print(" Exercise 7 - C\n")
print("The min value of magnitude is : ", np.min(magnitude))
print("The max value of magnitude is : ", np.max(magnitude))
print("The mean value of magnitude is : ", np.mean(magnitude))

# part d

print("----------------------------------------------")
print(" Exercise 7 - D\n")


def convert(dateString):
    return dateutil.parser.parse(dateString, dayfirst=True).year


# get only column of years from the file
years = np.genfromtxt('database.csv', delimiter=',', usecols=(0),
                      converters={0: convert}, dtype='unicode', skip_header=True)

# get only columns of earthquake
earth_quakes = np.genfromtxt('database.csv', converters={0: lambda x: parse(x)}, delimiter=',', usecols=(4),
                             dtype='unicode', skip_header=True)

# get the list of years (removing duplicate ones)
unique_years = np.unique(years, axis=0)

number = []
n = 0

for year in unique_years:
    for i in range(len(years)):
        if years[i] == year and earth_quakes[i] == 'Earthquake':
            n += 1
    number.append(n)
    n = 0  # reset the counter

# as an example we only plot 5 values

unique_years = unique_years[:10]
number = number[: 10]
result = dict(zip(unique_years, number))  # Creat dictionary from two list
print(result)

print(f"result[1965] = {result[1965]}")

print("# As an example we only consider 5 values per each for the plot of the graph")
plt.title('Number of Earthquakes vs Years')
plt.xlabel('Years')
plt.ylabel('Number of Earthquakes')
plt.bar(range(len(result)), list(result.values()), align='center')  # Plot the result
plt.xticks(range(len(result)), list(result.keys()))
plt.show()
