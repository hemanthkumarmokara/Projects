import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

iriss = pd.read_csv("D:\python by hemanth\python by hemanth 2\iris classification\IRIS.csv")
#print(iriss.head())
#print(iriss.describe())
#print("target labels are ",iriss["species"].unique())


import plotly.express as px
fig = px.scatter(iriss,x='sepal_width',y='sepal_length',color='species')
fig.show()
