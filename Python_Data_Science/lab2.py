import dateutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

exam_data = {'name': ['Anita', 'Adaleta', 'Lena', 'Ulrika', 'Mikael',
                      'Samuel', 'Simanthi', 'Kristian', 'Jusuf', 'Jonas'],
             'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
             'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
             'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no',
                         'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

# Exercise 1
data_df = pd.DataFrame(exam_data, index=labels)

# part a
print()
print('Exercise 1 : part A')
print(data_df)

# part b
print()
print('Exercise 1 : part B')
print('First three rows of the data frame:')
print(data_df.iloc[: 3, :])

# part c
print()
print('Exercise 1 : part C')
print(data_df.iloc[:, :2])
# print(data_df.loc[:, ['name', 'score']])

# part d
print()
print('Exercise 1 : part D')
print(data_df.loc[(data_df.attempts < 3) & (data_df.score > 14), :])

# part e
print()
print('Exercise 1 : part E')
print(data_df.fillna(0))

# Exercise 2

data_df = pd.read_csv("database.csv", converters={'Date': lambda s: dateutil.parser.parse(s).year})

# # part A
print()
print('Exercise 2 : part A\n')
print('Only column names : ')
print(data_df.columns)
print()
print('Shape of data frame : ')
print(data_df.shape)
print('Number of rows of data frame  : ')
print(data_df.shape[0])
print('Number of columns of data frame  : ')
print(data_df.shape[1])

# # part B
print()
print('Exercise 2 : part B\n')
my_columns = ['Date', 'Depth', 'Magnitude']
# data_df = pd.read_csv('database.csv', usecols=my_columns)
print(data_df[my_columns])

# part C
print()
print('Exercise 2 : part C\n')
print("The min value of magnitude is : ", data_df['Magnitude'].min())
print("The max value of magnitude is : ", data_df['Magnitude'].max())
print("The mean value of magnitude is : ", data_df['Magnitude'].mean())

# part D
print('Exercise 2 : part D\n')

only_earthquakes = data_df[data_df['Type'] == 'Earthquake']
data_earthquakes_number = only_earthquakes.groupby('Date')['Date'].count()

# As an example we get 10 rows from data from
data_earthquakes_number = data_earthquakes_number[:10]

data_earthquakes_number.plot(kind='bar')
plt.xlabel('Years')
#plt.show()

# Exercise 3
print('Exercise 3 : part A\n')

# read the newfile.txt which its columns have no headers
contributors = pd.read_csv('newfile.txt', delimiter='|', )

# assign name to each columns
contributors['id'] = contributors.iloc[:, 0]
contributors['name'] = contributors.iloc[:, 1]
contributors['date_of_change'] = contributors.iloc[:, 2].str.split(' ').str[1]  # take only date not time
contributors['number_of_lines'] = contributors.iloc[:, 3]

# part B
print()
print('Exercise 3 : part B\n')

# group by name and count each names contributions
groupByName = contributors.groupby('name')['name'].count().sort_values(ascending=False)
print(groupByName)

# part C
print()
print('Exercise 3 : part C\n')

# plot the bar graph for every name's contributions
groupByName.plot(kind='bar', title='Contributors\' Contributions')
plt.ylabel('Contributions')
#plt.show()

# Exercise 4

# part A
print()
print('Exercise 4 : part A\n')


# function that checks magnitude column and returns rank of earthquake
def rank_earthquake(magnitude):
    if magnitude <= 6.5:
        return 'Medium'
    elif (magnitude > 6.5) and (magnitude <= 7.5):
        return 'Strong'
    else:
        return 'Extreme'


# make a new column and put the rank value in it
data_df['Magnitude_desc'] = data_df['Magnitude'].apply(rank_earthquake)
# print(data_df.Magnitude_desc)

# part B
print()
print('Exercise 4 : part B\n')

# Make new column decade by using the years in date column
data_df['Decade'] = data_df['Date'].apply(lambda x: (np.floor(x / 10) * 10))
# print(data_df.Decade)
# part C
print()
print('Exercise 4 : part C\n')

# teachers given code
zz = data_df.groupby(['Decade', 'Magnitude_desc'])['Magnitude_desc'].aggregate('count').unstack()
# plt.ylabel('total earthquakes per year')
# plt.show()

# plotting the graph using pandas pivot table
print(data_df.pivot_table('Magnitude', index='Decade', columns='Magnitude_desc', aggfunc='count'))
# data_df.pivot_table('Magnitude', index='Decade', columns='Magnitude_desc', aggfunc='count').plot()
# plt.show()


