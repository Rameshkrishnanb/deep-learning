#!/usr/bin/env python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD


from keras.models import Sequential
from keras.layers import Dense

X,y=make_regression(n_samples=1000,n_features=20,noise=0.1,random_state=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


model=Sequential()
model.add(Dense(30,input_shape=(20,),activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(1,activation='linear'))


optimizer=SGD(lr=0.01,momentum=0.9)

model.compile(loss='mean_squared_logarithmic_error',optimizer=optimizer,metrics=['mse'])

model.fit(X_train,y_train,epochs=100,verbose=1,validation_data=(X_test,y_test))

results=model.evaluate(X_test,y_test)
print('Final test set loss:{:4f}'.format(results[0]))
print('Final test set accuracy:{:4f}'.format(results[0]))

