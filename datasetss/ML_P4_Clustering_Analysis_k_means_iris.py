import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


df = pd.read_csv('Iris.csv')
print(df.head())

x = df.iloc[:,[1,2,3,4]]

inertia = []
for i in range(1,11):
  k_mean = KMeans(n_clusters=i,max_iter=100,init="k-means++",n_init=10,random_state=42)
  k_mean.fit(x)
  inertia.append(k_mean.inertia_)


plt.plot(range(1,11),inertia,marker = '*')
plt.xlabel("No. of custer")
plt.ylabel("Inertia")
plt.show()


k = 3
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(x)
y_kmeans


