from math import *
from tkinter import filedialog

from matplotlib.pyplot import *
from scipy import *

# task 5

# def readfile():
#     try:
#         in_path = filedialog.askopenfilename(title="Locate the file kwh.dat")
#         nf = open(in_path, 'r')
#         lf = nf.readlines()
#         nf.close()
#         return lf
#
#     except IOError:
#         return -1
#
#
# lf = readfile()
# if lf != -1:
#     date = []  # first list for storing date values
#     kwh = []  # second list for storing kwh values
#     for i in range(len(lf)):
#         date.append(lf[i].split("  ")[
#                         0])  # split every single line using the 2-spaces delimeter, then store the first part in dates list
#         kwh.append(int(
#             lf[i].split("  ")[1]))  # convert the secont part of split into an integer, then append it to the kwh list
#     print(date)
#     print(kwh)
# else:
#     print('An error occured trying to read the file. check the location of file then try again.')


# task 8

strindberg_file = open('strindberg.txt', 'r')

total_number_of_words = 0.0
sum_words_total_text = 0.0

for line in strindberg_file.readlines():
    # put each line as one_line
    one_line = line

    # remove punctuation from each line
    one_line.replace('.', '')
    one_line.replace(',', '')
    one_line.replace('!', '')
    one_line.replace('?', '')

    # transform this line into a list of words from this line
    word_list = one_line.split()

    # length of words in each sentence
    word_length = [len(word) for word in word_list]

    # add sum of every word of each line
    sum_words_total_text += sum(word_length)

    # total number of words in one line

    total_number_of_words += len(word_length)

# compute average
average_word_total_text = sum_words_total_text / total_number_of_words
print(average_word_total_text)

strindberg_file.close()

# task 13


jonkoping_table = open('jonkoping.dat', 'r')
for line in jonkoping_table.readlines():
    date, time, temp = line.split(';')
    temp = float(temp)
    if temp > 0:
        print(f"The first frost came on {date} at {time} . The temperature recorded was {temp} C ")
    else:
        break

print(f" No frost occurred before {date} at {time}. There the temperature was {temp} C ")

