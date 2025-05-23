import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

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

x = taxi_df['hour']
y = taxi_df['price']
plt.title("time vs price scatter plot")
plt.xlabel("hour")
plt.ylabel("price")
plt.scatter(x, y, color='hotpink')
plt.show()

x=taxi_df['hour']
plt.hist(x)
plt.title("taxi journeys per hour")
plt.xlabel("hour")
plt.ylabel("count")
plt.show()

sns.lineplot(x=taxi_df['hour'], y=taxi_df['price'], data=taxi_df)
plt.show()

x = taxi_df['distance']
y = taxi_df['price']
plt.title("distance vs price scatter plot")
plt.xlabel("distance")
plt.ylabel("price")
plt.scatter(x, y, color='red')
plt.show()
