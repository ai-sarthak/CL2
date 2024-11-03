import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/content/AQI Data Set.csv')
df.head()

print(df.columns)
columns = ['id','months','PM10','SO2','NOx','PM2.5','Ammonia','O3','CO','Benzene','AQI']
df.columns = columns

print(df.head())

print(df.isna().sum())
df.dropna(inplace= True)
print(df.isna().sum())

print(df.describe())
print(df.info())

plt.figure(figsize=(12,6))
plt.plot(df['months'],df['AQI'])
plt.show()

plt.figure(figsize=(12,6))
plt.plot(df['months'],df['PM10'])
plt.show()

plt.figure(figsize=(12,6))
plt.plot(df['months'],df['SO2'])
plt.show()

plt.figure(figsize=(12,6),)
plt.plot(df['months'],df['NOx'],color='red')
plt.show()
