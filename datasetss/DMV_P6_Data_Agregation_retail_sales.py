import pandas as pd

df = pd.read_csv('/content/retail_sales_data.csv')
print(df.head())
print(df.describe())
print(df.info())

print(df.isna().sum())
print(df.isnull().sum())

df.drop(['invoice_no','customer_id','gender','age','payment_method'],inplace=True,axis=1)
print(df.head())

df['sales'] = df['quantity']*df['price']
print(df.head())

aggregate_sales = df.groupby('shopping_mall')['sales'].sum()
aggregate_sales.plot(kind='bar')
print(aggregate_sales)

print("Top Performing:",aggregate_sales.idxmax())


aggregate_sales = df.groupby(['shopping_mall','category'])['sales'].sum().unstack()
aggregate_sales.plot(kind='bar',stacked=True)
print(aggregate_sales)
