import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import seaborn as sns
# Data=mnist.load_data()
Data=pd.read_csv(r"D:\worldwide.csv")
WorkData=Data.head(n=100)
print(WorkData)
print(WorkData.info())
from sklearn.impute import SimpleImputer
pipline=Pipeline([('imputer',SimpleImputer(strategy="median"),('Attribute_Ader',StandardScaler()))])
# pipline.fit_transform(WorkData["Age"])
print(WorkData.info())
# X_train,X_test,y_train,y_test=train_test_split()
Data.hist(figsize=(15,25))
plt.show()
Data_neumirc_list=["Rank","Cost index"]
Data_neumirc_list_label=Data.loc[:,"Purchasing power index"]
Data_neumirc=Data.loc[:,Data_neumirc_list]
print(Data_neumirc)
Lin_mod=LinearRegression()
Lin_mod.fit(Data_neumirc_list,Data_neumirc_list_label)
some_Data=Data_neumirc_list_label.iloc[:5]