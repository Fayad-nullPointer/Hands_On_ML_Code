import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit 
from pandas.plotting import scatter_matrix
#Assume You want to Load Data From Wep But Your Data Change by Time 
#Apply This Code To Get Updated
# Download_link="https://raw.githubusercontent.com/ageron/handson-m12/master/"
# Data_path=os.path.join("Datasets","Name")
# Final_Url=Download_link+"path of file"
# #Note that You Will Download Compressed File.tgz And This Function Will Decmpress The File
# def Fetch_data(url=Final_Url,path=Data_path):
#     if not os.path.isdir(Data_path):
#         os.mkdir(Data_path)
#     tgz_path=os.path.join(Data_path,"Name.tgz")
#     urllib.request.urlretrieve(Final_Url,tgz_path)
#     File_Name_tgz=tarfile.open(tgz_path)
#     File_Name_tgz.extractall(path=Data_path)
#     File_Name_tgz.close()    
# # AlHmiudllah.Now You Have "Data_path" include Decompressed Data.csv
# def Load_Data_house(path=Data_path):
#     csv_path=os.path.join(Data_path,"Name.csv")
#     return pd.read_csv(csv_path)

# #if You Wnat To Make HistoGrph For Data 
# Load_Data_house.hist(bins=50,figsize=(20,15))    
# # Now Subbose You Want make Stratfild Shuffle split 
# #first You Should Determine Catigorial Attrbiuede To Make Sure That Train Data Will be Representitive
# Load_Data_house["New_name_Attrbuite"]=pd.cut(Load_Data_house["Catigorial Attrbiuede"],bins=[0,1.5,1.6,2],labels=[1,2,3])
# split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
# for train_index,test_index in split.split(Load_Data_house(),Load_Data_house["New_name_Attrbuite"]):
#     Star_train_test=Load_Data_house().loc[train_index]
#     Start_test_index=Load_Data_house().loc[test_index]
# #if You Want To Visualizing Scatter Data With Scatter
# Load_Data_house().plot(kind="scatter",x="x",y="y",alpha=0.4,s=Load_Data_house()[""],c=Load_Data_house()[""],colormap=plt.get_cmap("jet"),colorbar=True)
# # if you want To Make Corr Matrix To Find Relation Between 2 Variabls
# Attrbiuts=["",""]
# scatter_matrix(Load_Data_house()[Attrbiuts],figsize=(12,8))
# #To Make Clean Data From Null Cells
from sklearn.impute import SimpleImputer
data=pd.read_csv("D:\Countries.csv")
WorkData=data.head(n=100)
print(WorkData)
print(WorkData.isnull().sum())
print(WorkData.info())
WorkData["Ease of Doing Business"].fillna(WorkData["Ease of Doing Business"].median(),inplace=True)
print(WorkData.info())
#Imputer That is Way To Fill Missing Data
# impurter=SimpleImputer(strategy="median")
# z=WorkData.drop("Ease of Doing Business",axis=1)
# impurter.fit(z)
# impurter.statistics_
#Scince That ML Algorithm Deal With Neumircal Data We Will Decode Our Object Column
Country_Code=WorkData["Country Code"]
Country_Code.head(10)
# from sklearn.preprocessing import OrdinalEncoder
# O_E=OrdinalEncoder()
# Country_Code_Encoded=O_E.fit_transform(Country_Code)
# print(Country_Code_Encoded[:10].reshape(-1,1)) 
# print(O_E.categories_)
# from sklearn.preprocessing import OneHotEncoder
# One_Hot_Encoder=OneHotEncoder()
# Country_Code_Encoded2=One_Hot_Encoder.fit_transform(Country_Code)
# print(Country_Code_Encoded2.toarary())
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# You can add More Than One Transformer
List_To_drop=["Country Name","Country Code","Continent Name",]
WorkData.drop(WorkData.loc[:,List_To_drop],inplace=True,axis=1)
print(WorkData)
WorkData.plot(kind="scatter",x="GDP Per Capita",y="Population Density",figsize=(20,25))
plt.show()
WorkData.hist()
plt.show()
# pipline=Pipeline([('imputer',SimpleImputer(strategy="median"),('adder',StandardScaler()))])
# print(pipline.fit_transform(WorkData))
# from sklearn.compose import ColumnTransformer
# num_OF_Attributes=list(WorkData)
# Cat_attribuite=WorkData

