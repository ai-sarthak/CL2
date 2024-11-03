import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import category_encoders as ce

df = pd.read_csv('car_evaluation.csv')
print(df.head())

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y)

encoder = ce.OneHotEncoder(cols=['Buying price','Maintance cost','No of doors','No of persons','lug_boot','safety'])
x_train = encoder.fit_transform(x_train,y_train)
x_test = encoder.transform(x_test)

model = RandomForestClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))