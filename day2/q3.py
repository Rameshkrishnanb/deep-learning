#!/usr/bin/env python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

iris=load_iris()
X=iris.data
y_=iris.target.reshape(-1,1)

encoder=OneHotEncoder()
y=encoder.fit_transform(y_)

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2)
model=Sequential()
model.add(Dense(10,input_shape=(4,),activation='relu',name='fc1'))
model.add(Dense(10,activation='relu',name='fc2'))
model.add(Dense(3,input_shape=(4,),activation='softmax',name='output'))
optimizer=Adam(lr=0.001)
model.compile(optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(Xtrain,ytrain,verbose=1,batch_size=3,epochs=50)
results=model.evaluate(Xtrain,ytrain)
print(result[0])
