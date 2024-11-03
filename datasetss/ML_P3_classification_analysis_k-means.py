import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/content/Social_Network_Ads.csv')
print(df.head())

x = df.iloc[:,[2,3]]
y = df.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y)

standard = StandardScaler()
x_train_std  = standard.fit_transform(x_train,y_train)
x_test_std = standard.transform(x_test)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train_std,y_train)
y_pred = model.predict(x_test_std)

accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
error_rate = 1 - accuracy
print(error_rate)
precision = precision_score(y_test,y_pred)
print(precision)
recall = recall_score(y_test,y_pred)
print(recall)
