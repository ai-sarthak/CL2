import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv('telecom_churn.csv')
print(df.head())
print(df.isna().sum().sum())
print(df.duplicated().sum())

print(df.info())
print(df.describe())


df.drop(['telecom_partner','age', 'state','city','pincode', 'date_of_registration'], inplace=True,axis=1)
df.head()

le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df.head()

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y)

scalar = StandardScaler()
x_train_std = scalar.fit_transform(x_train,y_train)
x_test_std = scalar.transform(x_test)

print("****************************")
print(x_train_std)

df.to_csv('cleaned.csv')


