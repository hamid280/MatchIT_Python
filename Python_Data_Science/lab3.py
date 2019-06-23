import dateutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import conda
from itertools import chain
from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib

from mpl_toolkits.basemap import Basemap
import seaborn as sns

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
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='cyl', resolution=None,
            width=8E6, height=8E6,
            llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=-180, urcrnrlon=180, )
m.etopo(scale=0.5, alpha=0.5)

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

# Exercise 3

print()
print('Exercise 3 : part A\n')
df_nan_removed = df_only_earthquake.dropna()
print('Number of rows which contained NAN values and has been removed : '
      , len(df_only_earthquake) - len(df_nan_removed))

print()
print('Exercise 3 : part B\n')

sns.pairplot(df_nan_removed, hue='Type', height=1.5)
plt.show()

# Exercise 4
print()
print('Exercise 4 : part A\n')

df_years = data_df['Date']

# Since first parameter of linear regression must be 2D array
df_years = df_years[:, np.newaxis]

df_magnitude = data_df['Magnitude']

# check if the shapes are appropriate so that the regression could be applied
# print(df_years.shape)
# print(df_magnitude.shape)

# We should use the given formula to validate our prediction by considering the trained values
# we split 80% of the data to the training set while 20% of the data to test set using below code.
X_train, X_test, y_train, y_test = train_test_split(df_years, df_magnitude, test_size=0.2, random_state=0)

# After splitting the data into training and testing sets, Now we train our regression algorithm
reg = LinearRegression(fit_intercept=True)
reg.fit(X_train, y_train)  # training the algorithm

# The linear regression model basically finds the best value for the intercept and slope,
# which results in a line that best fits the data.

# To retrieve the intercept:
print('Simple regression model intercept value is ', reg.intercept_)

# For retrieving the slope:
print('Simple regression coefficient (Slope) is : ', reg.coef_)

print()
print('Exercise 4 : part B\n')

# Now that we have trained our algorithm, it’s time to make some predictions.
# To do so, we will use our test data and see how accurately our algorithm predicts
y_pred = reg.predict(X_test)

# (MSE) is the mean of the squared errors and is calculated as:
Mse = np.mean((y_pred - y_test) ** 2)
print('Mean Squared Error (MSEs) is :', Mse)

print()
print('Exercise 4 : part C\n')
plt.xlabel('Year')
plt.ylabel('Magnitude')

# plot the actual trained value
plt.scatter(X_test, y_test, color='blue')

# plot the predicated value
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

print()
print('Exercise 4 : part D\n')
print("As we can see in the plotted figure as well. The value of magnitude is clustered"
      "mainly between 5.5 and 6.0 through the years. Thus, our predicated line crossing between"
      "these two values makes logical sense and claims that our algorithm is correct.")

print()
print('Exercise 4 : part e\n')

df_depth = data_df['Depth']
df_depth = df_depth[:, np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(df_depth, df_magnitude, test_size=0.2, random_state=0)

reg = LinearRegression(fit_intercept=True)
reg.fit(X_train, y_train)  # training the algorithm

# To retrieve the intercept:
print('Simple regression model intercept value is ', reg.intercept_)

# For retrieving the slope:
print('Simple regression coefficient (Slope) is : ', reg.coef_)

y_pred = reg.predict(X_test)
Mse = np.mean((y_pred - y_test) ** 2)
print('Mean Squared Error (MSEs) for Depth vs Magnitude is :', Mse)
plt.xlabel('Depth')
plt.ylabel('Magnitude')

# plot the actual trained value
plt.scatter(X_test, y_test, color='blue')

# plot the predicated value
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

print()
print('Exercise 4 : part f\n')
print("As we can see in the plotted figure as well. The value of magnitude is clustered"
      "mainly between 5.5 and 6.0 through the given depths. Thus, our predicated line crossing between"
      "these two values makes logical sense and claims that our algorithm is correct.")


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
ax.scatter(xs=X_train['Latitude'], ys=X_train['Longitude'], zs=y_train, zdir='z', c='yellow')

# plot the predicted value as 3d
ax.scatter(xs=X_test['Latitude'], ys=X_test['Longitude'], zs=y_pred, zdir='z', c='r', marker='o')
plt.show()
