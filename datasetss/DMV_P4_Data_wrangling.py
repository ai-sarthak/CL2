import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv('Real-Estate dataset.csv')

print(df.head())
print(df.info())
print(df.describe())
print(df.isna().sum())
print(df.duplicated().sum())
df.drop_duplicates()
df.dropna()

encoder = LabelEncoder()
df['mainroad'] = encoder.fit_transform(df['mainroad'])
df['guestroom'] = encoder.fit_transform(df['guestroom'])
df['basement'] = encoder.fit_transform(df['basement'])
df['hotwaterheating'] = encoder.fit_transform(df['hotwaterheating'])
df['airconditioning'] = encoder.fit_transform(df['airconditioning'])
df['prefarea'] = encoder.fit_transform(df['prefarea'])
df['furnishingstatus'] = encoder.fit_transform(df['furnishingstatus'])

print(df.head())

#df.boxplot()

q1 = df['price'].quantile(0.25)
q3 = df['price'].quantile(0.75)
iqr = (q3 - q1)
min = q1 - (1.5*iqr)
max = q3 + (1.5*iqr)
df = df[(df['price'] > min) & (df['price']< max)]
print(df.head())

df.boxplot()

