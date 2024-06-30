import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
#Load Data
minst=fetch_openml('mnist_784',version=1,parser='auto')
#Discover Our Data
print(minst.keys()) 
print(minst["data"].shape)
print(minst["target"].shape)
X=minst["data"]
y=minst["target"]
Label=y[2]
print(Label)
# try:
#     X=minst["data"]
#     photo=X[2]
#     photo_reshaped=photo.reshape(28,28)
#     plt.imshow(photo_reshaped,cmap=mpl.cm.binary,interpolation="nearest")
#     plt.axis("off")
#     plt.show()
# finally:
#     print(NameError)
#Split Data -> Minst Split data 60000 train and 10000 test
X_train,X_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:]
#tarning binary Claasifir
y_train_5=(y_train==5)
y_test_5=(y_test==5)
sgd=SGDClassifier(random_state=42)
sgd.fit(X_train,y_train)
print(sgd.predict([X[4]]))
# #prformance using K-Fold On test book
# from sklearn.model_selection import StratifiedKFold
# from sklearn.base import clone
# for train_index ,test_index in 


#Start