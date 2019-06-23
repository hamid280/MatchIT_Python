import dateutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import conda
import seaborn as sns
from itertools import chain
from mpl_toolkits.basemap import Basemap
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib

# Exercise 1
print('Exercise 1 : part A\n')
# read only the requested columns from database.csv, convert the date columns and get only years
data_df = pd.read_csv("database.csv", usecols=['Date', 'Latitude', 'Longitude', 'Magnitude', 'Depth', 'Type'],
                      converters={'Date': lambda s: dateutil.parser.parse(s).year})
print(data_df[:5])
print()
print('Exercise 1 : part B\n')
print(data_df.pivot_table(columns='Type', aggfunc='size'))

# remove all rows which its column 'Type' contains other than 'Earthquake'
print()
print('Exercise 1 : part C\n')
df_only_earthquake = data_df[data_df['Type'].isin(['Earthquake'])]

# since we are removing rows, index has irregular values, thus we need to reset it.
df_only_earthquake.reset_index()
print(df_only_earthquake)


# Exercise 2


def draw_map(m, scale=0.2):
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)

    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)

    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')


print()
print('Exercise 2 : part B\n')

# cyl projection code from book
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='cyl', resolution=None,
            width=8E6, height=8E6,
            llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=-180, urcrnrlon=180, )
m.etopo(scale=0.5, alpha=0.5)

x_array = np.array(df_only_earthquake['Latitude'])
y_array = np.array(df_only_earthquake['Longitude'])

# Map every (long, lat) from x_array and y_array to (x, y) for plotting
x, y = m(x_array, y_array)
plt.plot(x, y, 'ok', markersize=2)
plt.show()

print()
print('Exercise 2 : part C\n')

# get only extreme earthquakes (Magnitude > 8)
df_severe_earthquake = df_only_earthquake[df_only_earthquake['Magnitude'] >= 8]

x_array = np.array(df_severe_earthquake['Latitude'])
y_array = np.array(df_severe_earthquake['Longitude'])

# Map every (long, lat) from x_array and y_array to (x, y) for plotting
x, y = m(x_array, y_array)

plt.plot(x, y, 'ok', markersize=2)
# plt.text(x, y, ' Seattle', fontsize=12);
plt.show()

print()
print('Exercise 2  : part D\n')
print("The min value of magnitude is : ", df_only_earthquake['Magnitude'].min())
print("The max value of magnitude is : ", df_only_earthquake['Magnitude'].max())
print("The mean value of magnitude is : ", df_only_earthquake['Magnitude'].mean())


# Exercise 4
print()
print('Exercise 4 : part A\n')

# Exercise 5

print()
print('Exercise 5 : part A\n')

X_train, X_test, y_train, y_test = train_test_split(data_df[['Latitude', 'Longitude']],
                                                    data_df['Magnitude'], test_size=0.2, random_state=0)

reg = LinearRegression(fit_intercept=True)
reg.fit(X_train, y_train)  # training the algorithm

# To retrieve the intercept:
print('Simple regression model intercept value is ', reg.intercept_)

# For retrieving the slope:
print('Simple regression coefficient (Slope) is : ', reg.coef_)

print()
print('Exercise 5 : part B\n')
y_pred = reg.predict(X_test)
Mse = np.mean((y_pred - y_test) ** 2)
print('Mean Squared Error (MSEs) for Latitude, Longitude vs Magnitude is :', Mse)

print()
print('Exercise 5 : part C\n')

# plot the actual value as 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Magnitude')

# plot the actual value as 3D
ax.scatter(xs=X_train['Latitude'], ys=X_train['Longitude'], zs=y_train, zdir='z', c='gray')

# plot the predicted value as 3d
ax.scatter(xs=X_test['Latitude'], ys=X_test['Longitude'], zs=y_pred, zdir='z', c='r', marker='o')
#plt.show()
