import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_csv = pd.read_csv('format1.csv')
df_excel = pd.read_excel('format2.xlsx')
df_json = pd.read_json('format3.json')

df = pd.concat([df_csv,df_excel,df_json])


print(df.duplicated().sum())
df.drop_duplicates()
print(df.isna().sum())
df.dropna()
df.fillna(0)

df.Date = pd.to_datetime(df.Date)
df.Time = pd.to_datetime(df.Time)

print(df.head())
print(df.describe())
print(df.info())
print(df.dtypes)

#df.boxplot()
#sns.pairplot(df)
plt.bar(df['Product line'],df['Total'])


plt.show()