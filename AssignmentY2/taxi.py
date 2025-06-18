import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


taxi_df = pd.read_csv('TaxiRideShare .csv', index_col=0)
print(taxi_df.head())
print(taxi_df.dtypes)
# convert into a correct format
taxi_df['datetime'] = pd.to_datetime(taxi_df['datetime'])
taxi_df['price'] = pd.to_numeric(taxi_df['price'], errors='coerce')         # error = 'coerce' replaces non-convertable values with NaN
print(taxi_df.dtypes)

# info about the data
print(taxi_df.info)
print( taxi_df.isna() .sum())
print("\nTotal number of duplicates")
print(taxi_df.duplicated() .sum())

# removing missing values
taxi_df.drop(['timestamp'], axis=1, inplace=True)
taxi_df.drop(['datetime'], axis=1, inplace=True)
taxi_df.drop(['visibility.1'], axis=1, inplace=True)
taxi_df.dropna(inplace=True)
print(taxi_df.isna().sum())
print(taxi_df.info)

# removing duplicates
print("\nTotal number of duplicates")
print(taxi_df.duplicated() .sum())
taxi_df.drop_duplicates(inplace=True)
print("\nTotal number of duplicates")
print(taxi_df.duplicated() .sum())
print(mpl.__version__)

#c = np.corrcoef(taxi_df['price'], taxi_df['hour'])
#print('Correlation between price and time\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['day'])
print('Correlation between price and day\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['month'])
print('Correlation between price and month\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['distance'])
print('Correlation between price and distance\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['surge_multiplier'])
print('Correlation between price and surge multiplier\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['temperature'])
print('Correlation between price and temp\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['apparentTemperature'])
print('Correlation between price and apparent temp\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['precipIntensity'])
print('Correlation between price and precip intensity\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['precipProbability'])
print('Correlation between price and precip probability\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['humidity'])
print('Correlation between price and humidity\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['windSpeed'])
print('Correlation between price and wind speed\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['windGust'])
print('Correlation between price and wind Gust\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['visibility'])
print('Correlation between price and visibilty\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['temperatureHigh'])
print('Correlation between price and temp high\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['temperatureLow'])
print('Correlation between price and temp low\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['dewPoint'])
print('Correlation between price and dew\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['pressure'])
print('Correlation between price and pressure\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['windBearing'])
print('Correlation between price and windBearing\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['cloudCover'])
print('Correlation between price and clouds\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['uvIndex'])
print('Correlation between price and UV\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['moonPhase'])
print('Correlation between price and moon\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['precipIntensityMax'])
print('Correlation between price and PIM\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['temperatureMin'])
print('Correlation between price and temp min\n',c)
c = np.corrcoef(taxi_df['price'], taxi_df['temperatureMax'])
print('Correlation between price and temp max\n',c)


x = taxi_df['hour']
y = taxi_df['price']
plt.title("time vs price scatter plot")
plt.xlabel("hour")
plt.ylabel("price")
plt.scatter(x, y, color='hotpink')
#plt.show()

x=taxi_df['hour']
plt.hist(x)
plt.title("taxi journeys per hour")
plt.xlabel("hour")
plt.ylabel("count")
#plt.show()

sns.lineplot(x=taxi_df['hour'], y=taxi_df['price'], data=taxi_df)
#plt.show()

x = taxi_df['distance']
y = taxi_df['price']
plt.title("distance vs price scatter plot")
plt.xlabel("distance")
plt.ylabel("price")
plt.scatter(x, y, color='red')
#plt.show()

x = taxi_df['surge_multiplier']
y = taxi_df['price']
plt.title("surge multiplier vs price scatter plot")
plt.xlabel("surge multiplier")
plt.ylabel("price")
plt.scatter(x, y, color='red')
#plt.show()
pd.set_option('display.max_columns', None)
print("Mode values:")
print(taxi_df.mode())
print("\n price standard deviation:")
print(taxi_df['price'].std())
print("\n distance standard deviation:")
print(taxi_df['distance'].std())
print("\n surge multiplier standard deviation:")
print(taxi_df['surge_multiplier'].std())
print("\n price skewness:")
print(taxi_df['price'].skew())
print("\n distance skewness:")
print(taxi_df['distance'].skew())
print("\n surge_multiplier skewness:")
print(taxi_df['surge_multiplier'].skew())

#seperate dependent and independent values
#independent variables
x = taxi_df.loc[:,['distance', 'surge_multiplier']].values
#print(x)
#dependent variable
y = taxi_df.loc[:,['price']].values
#print(y)
#splitting dataset into training and testing set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

#creating linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predictiong the test set
y_pred = regressor.predict(x_test)
#check accuracy
print('Coefficents: ', regressor.coef_)
#the mean squared error
print('Mean squared error: %.2f' % np.mean((regressor.predict(x_test) - y_test) ** 2))
#explained variance score: 1 is perfect predicition
print('Variance score: %.2f' % regressor.score(x_test, y_test))

#save model to disk
filename = 'finalised_model.sav'
pickle.dump(regressor, open(filename, 'wb'))