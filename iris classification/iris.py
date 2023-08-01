import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

iriss = pd.read_csv("D:\python by hemanth\python by hemanth 2\iris classification\IRIS.csv")
#print(iriss.head())
#print(iriss.describe())
#print("target labels are ",iriss["species"].unique())

"""
#plotting using plotly library
import plotly.express as px
fig = px.scatter(iriss,x='sepal_width',y='sepal_length',color='species')
fig.show()
"""

x = iriss.drop("species", axis=1)
y = iriss["species"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

#finally for prediction
x_new = np.array([[6, 3, 3, 1]])
prediction = knn.predict(x_new)
print("Prediction: {}".format(prediction))