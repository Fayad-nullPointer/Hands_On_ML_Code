import tensorflow as tf
import keras as k 
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
print(tf.__version__)
print(k.__version__)
fashon_mnist=k.datasets.fashion_mnist
(X_train_full,y_train_full),(X_test_full,y_test)=fashon_mnist.load_data()
print(X_train_full.shape)
print(X_train_full.dtype)
#Prepare valiadte (Hold _Out Validatdion)
X_valid,X_train=X_train_full[:5000]/255,X_train_full[5000:]/255
y_valid,y_train=y_train_full[:5000]/255,y_train_full[5000:]/255
plt.imshow(X_train_full[0])
# plt.show()
#Start Create neural network MLP this called Seq API of Keras
model=k.models.Sequential()
model.add(k.layers.Flatten(input_shape=[28,28]))
model.add(k.layers.Dense(300,activation="relu"))
model.add(k.layers.Dense(100,activation="relu"))
model.add(k.layers.Dense(10,activation="softmax"))
print(model.summary())
# print(model.layers)
#till now we have make model to init the parameter of model now let's compile model with loss function
#-----------------You can Also print weights for other Layers
model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
print("Done")
history=model.fit(X_train,y_train,epochs=3,validation_data=(X_valid,y_valid))
print(history)
import pandas as pd
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
#Drawing Trainig Loss and Accuracy Metric Above Line
#Builidin Rrgrssion Model
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
housing=fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split( 
 housing.data, housing.target) 
X_train, X_valid, y_train, y_valid = train_test_split( 
 X_train_full, y_train_full) 
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_valid_scaled = scaler.transform(X_valid) 
X_test_scaled = scaler.transform(X_test)
#Now let's Building Complex Model Using Keras Using Sup Classing API
class WideandDeepModel(k.models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



