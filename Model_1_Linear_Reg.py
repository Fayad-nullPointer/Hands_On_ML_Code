from sklearn.linear_model import LinearRegression
import matplotlib as plt
import numpy as np
import pandas as pd
#Load Data
oecd_bli=pd.read_csv("Path",thousands=',')
gdp_Per_capita=pd.read_csv("path",thousands=',',delimiter='\t' ,encoding='latin1',na_values="n/a")
# We Will Use A Function This Function Will Join 2 csv file from Diff Sources
def prepare_countery_state(file1,file2):
    file1=file1[file1["INEQULITY"]=="TOT"]
    #...Too Long And Not Spacific To ML
#There is Another Way To Make Linear Reg Model Call k_nearest Neighbors ((Instance Based Algorithm))
# the Method of Above 
import sklearn.neighbors
model=sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)

    