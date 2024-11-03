import requests
import pandas as pd
import matplotlib.pyplot as plt
import json

api_key = '91dc4e027d565d766d090eec6efb196c'
countries = ['Japan' , 'Saudi Arabia' , 'United States of America' , 'Indonesia' , 'India' , 'Egypt']
max_temp= []
min_temp = []
humidity= []
windspeed = []

for country in countries:
  data = requests.get(url=f'http://api.openweathermap.org/data/2.5/weather?q={country}&APPID={api_key}&units=metric')
  data = data.json()
  print(data)
  max_temp.append(data['main']['temp_max'])
  min_temp.append(data['main']['temp_min'])
  humidity.append(data['main']['humidity'])
  windspeed.append(data['wind']['speed'])

print(max_temp)
print(min_temp)
print(humidity)
print(windspeed)

df  = pd.DataFrame()
df['country'] = countries
df['max_temp'] = max_temp
df['min_temp'] = min_temp
df['humidity'] = humidity
df['windspeed'] = windspeed

df.head()

print(df.isna().sum())
print(df.duplicated().sum())

print(df.describe())
print(df.info())

plt.bar(x=df['country'],height=df['max_temp'])
plt.show()