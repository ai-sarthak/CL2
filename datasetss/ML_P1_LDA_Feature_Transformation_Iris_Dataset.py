import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

df = pd.read_csv('Iris.csv')

print(df.head())

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y)

lda = LinearDiscriminantAnalysis()
x_train_lda = lda.fit_transform(x_train,y_train)
x_test_lda = lda.transform(x_test)




model = LogisticRegression()
model.fit(x_train_lda,y_train)

y_pred = model.predict(x_test_lda)

print(accuracy_score(y_test,y_pred)*100)
print(confusion_matrix(y_test,y_pred))




