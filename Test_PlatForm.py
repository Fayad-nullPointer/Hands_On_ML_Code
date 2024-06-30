import numpy as np
import matplotlib.pyplot as plt
# x=[[1,2,3],[45,9,2]]
# y=[[5,6,7],[4,97,5]]
# z=[[2,4,5],[0,0,0]]
# x1=np.array(x)
# y1=np.array(y)
# z1=np.array(z)
# v=np.broadcast(x,y).ndim
# print(np.c_[x1,y1,z1])
# # d={'belal':22,'MO_essam':6}
# # print(**d)
# lis=[12,3,56,7,8,93,2,2,1,3,4,5]
# print(lis[:-1:])
from sklearn import datasets
iris=datasets.load_iris()
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
print(iris.keys())
X=iris["data"][:,3:] #petal length
y=(iris["target"]==2).astype(np.int32)
print(y.shape)
log_reg.fit(X,y)
X_new=np.linspace(0,3,1000).reshape(-1,1)
y_proba=log_reg.predict_proba(X_new)
plt.plot(X_new,y_proba[:,1],"g-",label="Iris_Virginica")
plt.plot(X_new,y_proba[:,0],"b--",label="NOT Iris_Virginica")
plt.show()
# from sklearn.svm import SVC
# from sklearn.svm import LinearSVC
# from sklearn.datasets import make_moons
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# SVm=Pipeline([
#     ("poly_featurs",PolynomialFeatures(degree=3)),
#     ("Scalar",StandardScaler()),
#     ("svm_clf",LinearSVC())    
# ]  
# )






