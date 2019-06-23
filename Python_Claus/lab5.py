from matplotlib.pyplot import *
from scipy import *
from numpy import *

# task 1

newfile = open('newfile.dat', 'w')
newfile.write('1 st line \n')
newfile.write('2nd line \n')
newfile.close()
file = open('newfile.dat', 'r')

# for line in file.readlines():
# print(line)

file.close()


# task 2
def tesfile():
    file = open('newfile.dat', 'r')
    return file


text1 = list(tesfile())
text2 = list(tesfile())

# task 3


# read the kwh.dat file as read only
kwh_file = open('kwh.dat', 'r')

# read every line in the file line by line
content = kwh_file.readlines()
kwh_file.close()

# put each line in the list
L_content = [line.split('  ') for line in content]

# store date separately in a new list
yearmonthday = [date[0] for date in L_content]

# store energy separately in a new list
kwh = [int(kwh[1]) for kwh in L_content]

# task 4
# yearmonthday.reverse()
# kwh.reverse()

# task 5

# first we convert the kwh list to array so that we could use diff function
kwh_array = asarray(kwh)

# we use numpy.diff to find the difference between each element of the array
# print("###################################################")
diff_list = diff(kwh_array)

print(diff_list)

# task6


# plot(yearmonthday, kwh)
# show()

# task7

# first we find the maximum energy in the list kwh

max_energy = max(kwh)
index = 0

# search for the index that max energy occurs
for i in range(len(kwh)):
    if kwh[i] == max_energy:
        index = i

max_energy_month2 = yearmonthday[index]

# much simpler solution by using argmax comment
max_index = argmax(kwh)
max_energy_month = yearmonthday[int(max_index)]

# month with the smallest energy

min_index = argmin(kwh)
min_energy_month = yearmonthday[int(min_index)]

# task 8

diff_max_index = argmax(diff_list)
diff_min_index = argmin(diff_list)

print(f" largest increase consumption with respect to the previous {yearmonthday[int(diff_max_index)]}")
print(f" smallest increase consumption with respect to the previous {yearmonthday[int(diff_min_index)]}")

# task 9

mean_value = sum(kwh) / len(kwh)

print(mean_value)

#task 13

