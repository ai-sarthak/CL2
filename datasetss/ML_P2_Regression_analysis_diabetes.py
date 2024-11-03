import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report, r2_score
from sklearn.linear_model import LinearRegression,LogisticRegression


df = pd.read_csv('diabetes.csv')

print(df.head())

print(df.describe())

df_info = df.describe()
#print(df_info.columns)

print("\n\n______________Standard:________________\n\n")
print(df.std())


print("\n\n______________VARIANCE:________________\n\n")
print(df.var())

print("\n\n______________Mode:______________________\n\n")
print(df.mode())

print("\n\n______________mean:______________________\n\n")
print(df.mean())

print("\n\n______________median:______________________\n\n")
print(df.median())

print("\n\n______________skew:______________________\n\n")
print(df.skew())

print("\n\n______________kurtosis:______________________\n\n")
print(df.kurtosis())




x = df.iloc[:,:-1]
y = df.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(x,y)

model_linear = LinearRegression()
model_linear.fit(x_train,y_train)
y_pred_linear = model_linear.predict(x_test)
print(r2_score(y_test,y_pred_linear)) # r2_score = 1-((sum(y_true - y_pred))^2 / (sum(y_true - y))^2)
                                      # r2_score is calculated since linear regression gives continuous value and y_test is binary value no accuracy can be calculated.

model_logistic = LogisticRegression()
model_logistic.fit(x_train,y_train)
y_pred_logistic = model_logistic.predict(x_test)
print(accuracy_score(y_test,y_pred_logistic))

