import numpy as np
import datetime
import pytz
import dateutil.parser
from collections import Counter
from dateutil import parser
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt


def convert(dateString):
    return dateutil.parser.parse(dateString, dayfirst=True).year


years = np.genfromtxt('database.csv', delimiter=',', usecols=(0),
                      converters={0: convert}, dtype=[('Date', 'i4'), ('Earthquake', 'U10')],
                      skip_header=True)
earth_quakes = np.genfromtxt('database.csv', converters={0: lambda x: parser.parse(x)}, delimiter=',', usecols=(4),
                             dtype='unicode', skip_header=True)
print(len(years))

# get the list of years (removing duplicate ones)
unique_years = np.unique(years, axis=0)
print(len(unique_years))

number = []
n = 0

for year in unique_years:
    for i in range(len(years)):
        if years[i] == year and earth_quakes[i] == 'Earthquake':
            n += 1
    number.append(n)
    n = 0  # reset the number

result = dict(zip(unique_years, number))  # Creat dictionary from two list

plt.title('Number of Earthquakes vs Years')
plt.xlabel('Years')
plt.ylabel('Number of Earthquakes')
plt.bar(range(len(result)), list(result.values()), align='center')  # Plot the result
plt.xticks(range(len(result)), list(result.keys()))
plt.show()
